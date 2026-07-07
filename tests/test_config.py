from ml_collections import ConfigDict

from action_bridge.config import apply_overrides, available_config_names, load_config, to_plain_dict


def test_python_config_loads_as_config_dict():
    assert "toy_delayed_continuous" in available_config_names()
    config = load_config("toy_delayed_continuous")
    assert isinstance(config, ConfigDict)
    assert config.config_name == "toy_delayed_continuous"
    assert config.model.latent_type == "continuous"


def test_overrides_preserve_nested_config_dicts():
    config = apply_overrides(
        load_config("toy_delayed_categorical"),
        ["device=cpu", "optim.max_steps=10", "logging.wandb.enabled=true"],
    )
    assert config.device == "cpu"
    assert config.optim.max_steps == 10
    assert config.logging.wandb.enabled is True
    assert isinstance(to_plain_dict(config)["optim"], dict)
