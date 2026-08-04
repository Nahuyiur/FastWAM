from types import SimpleNamespace

import pytest

from fast_wam.train.optimizer_contract import all_parameter_adamw_config


def test_all_parameter_adamw_removes_megatron_weight_decay_exemptions():
    config = SimpleNamespace(decoupled_weight_decay=True)

    def original(args):
        assert args == "args"
        return config, {"bias_and_norm": {"wd_mult": 0.0}}

    actual_config, overrides = all_parameter_adamw_config(original)("args")

    assert actual_config is config
    assert overrides == {}


def test_all_parameter_adamw_rejects_coupled_l2_decay():
    config = SimpleNamespace(decoupled_weight_decay=False)

    def original(args):
        return config, {}

    with pytest.raises(ValueError, match="requires AdamW"):
        all_parameter_adamw_config(original)(None)
