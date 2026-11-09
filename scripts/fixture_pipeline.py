"""Real CPU acceptance: parse/window/split/cache/train/resume/reload/MIDI/render.

Uses only synthetic fixtures and an explicit new workspace. Not a quality score.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_script(name):
    spec = importlib.util.spec_from_file_location('fixture_script', ROOT / 'scripts' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(workspace):
    from open_fret.alphatex import AlphaTexParser, PARSER_VERSION
    from open_fret.sequences import SequenceBuilder, AlignedWindower
    from dataclasses import asdict
    import pretty_midi
    root = Path(workspace).resolve()
    if root.exists():
        raise FileExistsError(f"Fixture workspace exists: {root}; use a new directory")
    root.mkdir(parents=True)
    data = root / 'individual'
    data.mkdir()
    # Ten different source melodies; all tokens are a predefined fixture domain,
    # not harvested from validation/test target annotations.
    for index in range(10):
        track = AlphaTexParser().parse_text(
            '\\track\n\\tuning E5 B4 G4 D4 A3 E3\n:4 ' + ' '.join(f'{(index+j)%12}.1' for j in range(3)))[0]
        steps, tuning = SequenceBuilder().build_steps(track)
        for w in AlignedWindower(64, 64).window(steps, tuning, source=f'fixture_{index}.tex'):
            payload = asdict(w)
            payload.update(midi=w.midi_tokens, tab=w.tab_tokens, parser_version=PARSER_VERSION,
                           track=w.track_index, window=w.window_index)
            (data / f'song{index}__c0.json').write_text(json.dumps(payload))
    load_script('03_split.py').split_dataset(str(data), str(root / 'splits'))
    words = ([f'TAB_{s}_{f}' for s in range(1, 7) for f in range(25)] +
             [f'{kind}_{p}' for kind in ('NO', 'NF') for p in range(128)] +
             ['CAPO_0', 'TUNING_64_59_55_50_45_40', 'TS_120', 'TS_240', 'TS_480'])
    (root / 'vocab.json').write_text(json.dumps(words))
    load_script('02_tokenize.py').build_tokenizer(str(root / 'vocab.json'), str(root / 'tokenizer'))
    load_script('04_cache_dataset.py').create_cache(
        tokenizer_dir=str(root / 'tokenizer'), data_dir=str(data),
        train_list=str(root / 'splits/train_files.txt'), val_list=str(root / 'splits/val_files.txt'),
        test_list=str(root / 'splits/test_files.txt'), cache_dir=str(root / 'cache'), max_length=64, require_test=True)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    command = [sys.executable, 'scripts/train_model.py', '--output-dir', str(root / 'run'),
               '--tokenizer-dir', str(root / 'tokenizer'), '--cache-dir', str(root / 'cache'),
               '--workers', '0', '--batch-size', '1', '--grad-accum', '1', '--sample-songs', '0',
               '--eval-steps', '1', '--save-steps', '1', '--warmup-steps', '0']
    subprocess.run(command + ['--max-steps', '1'], cwd=ROOT, env=env, check=True)
    subprocess.run(command + ['--resume', '--max-steps', '2'], cwd=ROOT, env=env, check=True)
    pm = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=480)
    instrument = pretty_midi.Instrument(24)
    instrument.notes = [pretty_midi.Note(100, 64, .125 + i*.25, .25 + i*.25) for i in range(12)]
    pm.instruments.append(instrument)
    midi = root / 'input.mid'
    pm.write(str(midi))
    subprocess.run([sys.executable, 'scripts/inference.py', '--input', str(midi),
                    '--model', str(root / 'run/final'), '--max-input-tokens', '12', '--max-new-tokens', '10',
                    '--out', str(root / 'corrected.tokens'), '--raw-out', str(root / 'raw.tokens'),
                    '--report', str(root / 'inference.json'), '--tex-out', str(root / 'output.tex'),
                    '--ascii-out', str(root / 'output.txt')], cwd=ROOT, env=env, check=True)
    result = json.loads((root / 'inference.json').read_text())
    actual = json.loads(subprocess.check_output(['node', str(ROOT / 'tools/verify_score.cjs'),
                                                str(root / 'output.tex')], text=True))
    expected = [{k: n[k] for k in ('start', 'end', 'pitch')} for n in result['source_notes']]
    sort = lambda notes: sorted((n['start'], n['end'], n['pitch']) for n in notes)
    assert sort(actual) == sort(expected), (actual, expected)
    assert len(result['windows']) > 1
    assert result['raw_metrics']['timed_pitch']['f1'] == 1
    assert json.loads((root / 'run/run.json').read_text())['result']['global_steps'] == 2
    # A separately generated browser fixture: original training previews untouched.
    from open_fret.viewer import build_view_bundle, to_html
    bundle = build_view_bundle(midi_path=str(midi), tab_tokens=result['corrected_tokens'],
                               reference_tab_tokens=result['corrected_tokens'], title='CPU fixture acceptance')
    bundle['alpha_tex'] = (root / 'output.tex').read_text()
    bundle['expected_notes'] = expected
    dist = (ROOT / 'ui/tab_data_viewer/dist/tab-data-viewer.umd.js').read_text()
    (root / 'viewer.html').write_text(to_html(bundle, dist_js=dist))
    (root / 'acceptance.json').write_text(json.dumps({'status': 'complete', 'schema_version': 1,
                                                    'notes': len(actual), 'windows': len(result['windows']),
                                                    'scope': 'synthetic CPU plumbing; not model quality'}, indent=2))
    print(f'CPU FIXTURE PASSED: {root}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', required=True)
    main(ap.parse_args().workspace)
