import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hpo_optuna import LossHistoryCallback


class Trial:
    def __init__(self):
        self.user_attrs = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value

    def report(self, *args, **kwargs):
        pass

    def should_prune(self):
        return False


def test_all_components_persist_as_curves(tmp_path):
    trial = Trial()
    callback = LossHistoryCallback(trial, str(tmp_path / "components.json"))
    metrics = {"loss": 1.5, "ce_loss": 2, "pitch_loss": 1, "weighted_ce_loss": 0.6,
               "weighted_pitch_loss": 1, "equivalence_smoothing_delta": -0.1,
               "weighted_smoothing_delta": -0.03}
    callback.on_log(None, SimpleNamespace(global_step=50), None, logs=metrics)
    callback.on_evaluate(None, SimpleNamespace(global_step=50), None,
                         metrics={"eval_" + k: v for k, v in metrics.items()})
    saved = json.loads((tmp_path / "components.json").read_text())
    assert saved["train"][0]["weighted_ce_loss"] == 0.6
    assert saved["eval"][0]["eval_weighted_smoothing_delta"] == -0.03
    assert trial.user_attrs["eval_component_curve"][0]["eval_pitch_loss"] == 1
