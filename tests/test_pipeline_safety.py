import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from open_fret.artifacts import run_stage


def script(name):
    spec = importlib.util.spec_from_file_location('pipeline_test', ROOT / 'scripts' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stages_require_completion_content_and_input_identity(tmp_path):
    source = tmp_path / 'source'
    source.write_text('input')
    output = tmp_path / 'stage'
    def command(out):
        return [sys.executable, '-c', "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')", str(out / 'data')]
    run_stage(output, [source], command)
    run_stage(output, [source], command)
    source.write_text('changed')
    with pytest.raises(ValueError, match='Stale/corrupt'):
        run_stage(output, [source], command)
    source.write_text('input')
    (output / 'data').write_text('corrupt')
    with pytest.raises(ValueError, match='Stale/corrupt'):
        run_stage(output, [source], command)
    partial = tmp_path / 'partial'
    partial.mkdir()
    with pytest.raises(ValueError, match='Partial/unversioned'):
        run_stage(partial, [source], command)


def test_failed_stage_never_publishes_completion(tmp_path):
    source = tmp_path / 'source'
    source.write_text('x')
    output = tmp_path / 'stage'
    with pytest.raises(subprocess.CalledProcessError):
        run_stage(output, [source], lambda out: [sys.executable, '-c', 'raise RuntimeError("failure")'])
    assert not output.exists()
    assert source.read_text() == 'x'


def test_bad_empty_and_unsafe_archives_fail_atomically(tmp_path):
    extract = script('00_download.py').extract_archive
    archive, destination = tmp_path / 'data.zip', tmp_path / 'extracted'
    archive.write_text('partial download')
    with pytest.raises(zipfile.BadZipFile):
        extract(archive, destination)
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../outside.tex', 'unsafe')
    with pytest.raises(ValueError, match='Unsafe'):
        extract(archive, destination)
    assert not (tmp_path / 'outside.tex').exists()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('readme.txt', 'no scores')
    with pytest.raises(ValueError, match='no AlphaTex'):
        extract(archive, destination)
    assert not destination.exists()


def test_download_failure_propagates(monkeypatch, tmp_path):
    import huggingface_hub
    def fail(**kwargs):
        raise OSError('offline fixture')
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fail)
    with pytest.raises(OSError, match='offline'):
        script('00_download.py').download_and_save_dataset(tmp_path)


def test_cache_bad_json_empty_split_and_oversize_fail_without_output(tmp_path):
    from test_loss_cli import load_script
    from open_fret.alphatex import PARSER_VERSION
    cache = load_script('04_cache_dataset.py').create_cache
    data = tmp_path / 'data'
    data.mkdir()
    vocab = tmp_path / 'vocab.json'
    midi = ['CAPO_0', 'TUNING_64_59', 'NO_64', 'TS_480', 'NF_64']
    tab = ['TAB_1_0', 'TS_480']
    vocab.write_text(json.dumps(midi + tab))
    load_script('02_tokenize.py').build_tokenizer(str(vocab), str(tmp_path / 'tokenizer'))
    split = tmp_path / 'files.txt'
    split.write_text('')
    kwargs = dict(tokenizer_dir=str(tmp_path / 'tokenizer'), data_dir=str(data),
                  train_list=str(split), val_list=str(split), test_list=str(tmp_path / 'missing'),
                  cache_dir=str(tmp_path / 'cache'))
    with pytest.raises(ValueError, match='Empty'):
        cache(**kwargs)
    split.write_text('bad.json')
    (data / 'bad.json').write_text('{bad')
    with pytest.raises(Exception):
        cache(**kwargs)
    assert not (tmp_path / 'cache').exists()
    (data / 'bad.json').write_text(json.dumps(dict(midi=midi, tab=tab, parser_version=PARSER_VERSION)))
    with pytest.raises(Exception):
        cache(**kwargs, max_length=3)
    assert not (tmp_path / 'cache').exists()
