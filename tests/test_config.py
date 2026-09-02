"""Config-system tests.

The property that matters: a config file is the complete description of an
experiment. A typo must fail loudly rather than silently leaving a default in
place, because a silently-defaulted ablation is recorded in ``results/`` as
the ablation it was not.
"""

from __future__ import annotations

import pytest

from flowconx.experiment import (
    ExperimentConfig,
    _parse_override,
    deep_update,
    from_dict,
    load_config,
    save_config,
)


def test_defaults_validate():
    ExperimentConfig().validate()


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown config key"):
        from_dict({"model": {"fusionn": "concat"}})
    with pytest.raises(ValueError, match="Unknown config key"):
        from_dict({"trian": {"epochs": 1}})


@pytest.mark.parametrize("fusion", ["cross_attention", "concat", "gated_sum", "late"])
def test_every_fusion_mode_validates(fusion):
    from_dict({"model": {"fusion": fusion}}).validate()


def test_invalid_enum_is_rejected():
    with pytest.raises(ValueError, match="model.fusion"):
        from_dict({"model": {"fusion": "magic"}})
    with pytest.raises(ValueError, match="classify_from"):
        from_dict({"model": {"classify_from": "z_nonexistent"}})


def test_inconsistent_combinations_are_rejected():
    # No second encoder to fuse with.
    with pytest.raises(ValueError, match="dual_encoder"):
        from_dict({"model": {"dual_encoder": False}})
    # Classifying from an embedding the configuration does not produce.
    with pytest.raises(ValueError, match="z_network"):
        from_dict({"model": {"dual_encoder": False, "fusion": "none", "classify_from": "z_network"}})
    # Stage 1 cannot be the whole run in a two-stage schedule.
    with pytest.raises(ValueError, match="stage1_epochs"):
        from_dict({"train": {"schedule": "two_stage", "epochs": 5, "stage1_epochs": 5}})
    # Observing more packets than were read from the CSV.
    with pytest.raises(ValueError, match="observed_packets"):
        from_dict({"data": {"max_packets": 10, "observed_packets": 20}})


def test_hash_ignores_seed_but_not_content():
    a = from_dict({"seed": 0})
    b = from_dict({"seed": 99})
    assert a.hash() == b.hash(), "seed must not change the experiment identity"
    c = from_dict({"seed": 0, "loss": {"lambda_adversarial": 0.0}})
    assert a.hash() != c.hash(), "a loss weight change must change the hash"


def test_run_dir_separates_seeds_and_splits():
    a = from_dict({"name": "x", "seed": 1})
    b = from_dict({"name": "x", "seed": 2})
    c = from_dict({"name": "x", "seed": 1, "data": {"split_protocol": "random_flow"}})
    assert a.run_dir != b.run_dir
    assert a.run_dir != c.run_dir


def test_override_parsing():
    assert _parse_override("model.fusion=concat") == {"model": {"fusion": "concat"}}
    assert _parse_override("loss.lambda_adversarial=0.5") == {"loss": {"lambda_adversarial": 0.5}}
    assert _parse_override("model.dual_encoder=false") == {"model": {"dual_encoder": False}}
    assert _parse_override("data.unknown_apps=[zoom, netflix]") == {"data": {"unknown_apps": ["zoom", "netflix"]}}
    with pytest.raises(ValueError):
        _parse_override("nonsense")


def test_deep_update_does_not_clobber_siblings():
    base = {"model": {"fusion": "cross_attention", "dropout": 0.1}}
    assert deep_update(base, {"model": {"dropout": 0.3}}) == {
        "model": {"fusion": "cross_attention", "dropout": 0.3}
    }


def test_shipped_configs_load(tmp_path):
    from pathlib import Path

    for path in sorted(Path("configs").glob("*.yaml")):
        config = load_config(path)
        config.validate()
        # Round-trip: saving and reloading must not change the experiment.
        saved = save_config(config, tmp_path / f"{path.stem}.yaml")
        assert load_config(saved).hash() == config.hash()


def test_defaults_chain_is_applied():
    smoke = load_config("configs/smoke.yaml")
    base = load_config("configs/base.yaml")
    assert smoke.data.limit == 3000, "smoke.yaml's own key must win"
    assert smoke.model.fusion == base.model.fusion, "inherited keys must come from base.yaml"


def test_cli_override_wins_over_file():
    config = load_config("configs/smoke.yaml", ["model.fusion=concat", "train.epochs=7"])
    assert config.model.fusion == "concat"
    assert config.train.epochs == 7


def test_defaults_chain_is_resolved_recursively():
    """Two-level inheritance must carry the grandparent's keys.

    The ablation configs inherit from ablation_base.yaml, which inherits from
    cesnet_main.yaml, which inherits from base.yaml. A single-level resolver
    silently drops the grandparent and leaves dataclass defaults in place, so
    an ablation would run on a different dataset and split than the reference
    row it is tabulated against, and nothing would report the mismatch.
    """
    ablation = load_config("configs/ablations/no_flow_metric.yaml")
    main = load_config("configs/cesnet_main.yaml")
    # From the grandparent, two levels up.
    assert ablation.data.dataset == main.data.dataset
    assert ablation.data.split_protocol == main.data.split_protocol
    assert ablation.data.csv == main.data.csv
    # From the parent.
    assert ablation.data.limit == 80000
    assert ablation.train.epochs == 10
    # Its own key.
    assert ablation.loss.lambda_flow_supcon == 0.0


def test_ablation_family_shares_one_budget():
    """Every ablation row must be comparable to every other row.

    A table whose rows ran at different budgets is not an ablation table.
    """
    from pathlib import Path

    reference = load_config("configs/ablation_base.yaml")
    for path in sorted(Path("configs/ablations").glob("*.yaml")):
        config = load_config(path)
        assert config.data.limit == reference.data.limit, f"{path.name} has a different data budget"
        assert config.data.dataset == reference.data.dataset, f"{path.name} uses a different dataset"
        assert config.data.split_protocol == reference.data.split_protocol, f"{path.name} uses a different split"
        # The input-budget sweep is the one family that legitimately varies
        # epochs indirectly; it varies observed_packets, never the schedule.
        if not path.stem.startswith("packets_"):
            assert config.train.epochs == reference.train.epochs, f"{path.name} has a different epoch budget"


def test_circular_defaults_are_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("defaults: [b.yaml]\nname: a\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("defaults: [a.yaml]\nname: b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")
