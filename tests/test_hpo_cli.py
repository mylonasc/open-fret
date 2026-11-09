"""Actual persistent Optuna/Trainer integration, synthetic CPU data only."""
import json
import os
from pathlib import Path
import subprocess
import sys

import optuna
from transformers import T5ForConditionalGeneration, AutoTokenizer
from test_loss_cli import fixture_cache

ROOT = Path(__file__).resolve().parents[1]


def test_persistent_trials_pruning_export_and_retraining(fixture_cache):
    out = fixture_cache / 'hpo'
    storage = f'sqlite:///{fixture_cache / "study.db"}'
    command = [sys.executable, str(ROOT / 'scripts/hpo_optuna.py'),
               '--fixture-space', '--study-name', 'cpu-fixture', '--storage', storage,
               '--tokenizer-dir', str(fixture_cache / 'tokenizer'),
               '--cache-dir', str(fixture_cache / 'cache'), '--output-root', str(out),
               '--max-steps', '2', '--eval-steps', '1', '--time-limit', '60',
               '--gpu-monitor-interval', '0']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    first = subprocess.run(command + ['--trials', '2', '--pruner', 'none'], env=env, cwd=ROOT,
                           capture_output=True, text=True, timeout=90)
    assert first.returncode == 0, first.stdout + first.stderr
    study = optuna.load_study(study_name='cpu-fixture', storage=storage)
    assert len(study.trials) == 2
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert study.trials[0].params != study.trials[1].params
    assert all(t.intermediate_values and t.user_attrs['seed'] == 42 + t.number for t in study.trials)
    best = json.loads((out / 'best_config.json').read_text())
    winner = out / f'trial_{study.best_trial.number}' / 'final'
    assert AutoTokenizer.from_pretrained(winner)
    assert T5ForConditionalGeneration.from_pretrained(winner).config.vocab_size > 0
    # Resume the same DB and exercise real pruning from Trainer's eval callback.
    resumed = subprocess.run(command + ['--trials', '1', '--pruner', 'threshold',
                                         '--prune-threshold', '0', '--pruner-warmup-steps', '0'],
                              env=env, cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    study = optuna.load_study(study_name='cpu-fixture', storage=storage)
    assert len(study.trials) == 3
    assert study.trials[-1].state == optuna.trial.TrialState.PRUNED
    assert study.trials[-1].intermediate_values
    summary = json.loads((out / 'study_summary.json').read_text())
    assert summary['direction'] == 'minimize'
    assert summary['data_protocol']['validation_fingerprint']
    # Export is consumed directly by the normal training CLI, without translating flags.
    retrain = subprocess.run([sys.executable, str(ROOT / 'scripts/train_model.py'),
                              '--hpo-config', str(out / 'best_config.json'),
                              '--tokenizer-dir', str(fixture_cache / 'tokenizer'),
                              '--cache-dir', str(fixture_cache / 'cache'), '--output-dir', str(out / 'retrain'),
                              '--max-steps', '1', '--sample-songs', '0', '--workers', '0',
                              '--eval-steps', '1', '--save-steps', '1'],
                             env=env, cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert retrain.returncode == 0, retrain.stdout + retrain.stderr
    manifest = json.loads((out / 'retrain/run.json').read_text())
    assert manifest['hyperparams']['lr'] == best['training_args']['lr']
    assert manifest['result']['global_steps'] == 1
