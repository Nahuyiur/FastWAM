from pathlib import Path
import importlib.util


SCRIPT = Path(__file__).parents[1] / "scripts" / "sync_megatron_train_log_to_wandb.py"
SPEC = importlib.util.spec_from_file_location("sync_megatron_train_log_to_wandb", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_iteration_line_extracts_training_metrics():
    line = (
        " [2026-08-03 13:38:19.881514] iteration     3100/   50000 | "
        "consumed samples:        99200 | elapsed time per iteration (ms): 3772.1 | "
        "learning rate: 4.998051E-05 | global batch size:    32 | loss: 2.947306E-01 | "
        "video loss: 2.282371E-01 | action loss: 6.649351E-02 | loss scale: 1.0 | "
        "grad norm: 0.601 | number of skipped iterations:   0 | "
        "number of nan iterations:   0 |\n"
    )

    step, payload = MODULE.parse_iteration_line(line)

    assert step == 3100
    assert payload["train/iteration"] == 3100
    assert payload["train/consumed_samples"] == 99200
    assert payload["train/global_batch_size"] == 32
    assert payload["train/loss"] == 0.2947306
    assert payload["train/video_loss"] == 0.2282371
    assert payload["train/action_loss"] == 0.06649351
    assert payload["performance/iteration_time_ms"] == 3772.1
    assert payload["train/skipped_iterations"] == 0
    assert payload["train/nan_iterations"] == 0


def test_parse_iteration_line_rejects_unrelated_output():
    assert MODULE.parse_iteration_line("Number of parameters: 3.78\n") is None


def test_sidecar_parser_keeps_latest_duplicate_iteration(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        " [2026-08-04 00:00:00] iteration 20/ 50000 | loss: 1.0 |\n"
        " [2026-08-04 00:00:01] iteration 20/ 50000 | loss: 0.5 |\n"
    )

    rows = MODULE.parse_log(log)

    assert len(rows) == 1
    assert rows[0][0] == 20
    assert rows[0][1]["train/loss"] == 0.5
