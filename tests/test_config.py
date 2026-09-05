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
    # Use a weight that differs from the default. An earlier version of this
    # test used 0.0, which silently became a no-op when the adversarial head
    # was defaulted off -- the assertion then compared a config against itself.
    c = from_dict({"seed": 0, "loss": {"lambda_prototype": 0.99}})
    assert c.loss.lambda_prototype != ExperimentConfig().loss.lambda_prototype
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
    assert ablation.data.subsample_rows == 72000
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


def test_limit_and_subsample_are_mutually_exclusive():
    """`limit` reads a biased prefix; `subsample_rows` is stratified.

    Setting both silently applies the prefix first, so the "stratified"
    subsample would be drawn from the earliest capture days only. The config
    refuses rather than producing a quietly biased experiment.
    """
    with pytest.raises(ValueError, match="Pick one"):
        from_dict({"data": {"limit": 1000, "subsample_rows": 500}})


def test_ablations_use_a_stratified_subsample_not_a_prefix():
    from pathlib import Path

    for path in sorted(Path("configs/ablations").glob("*.yaml")):
        config = load_config(path)
        assert config.data.limit is None, (
            f"{path.name} uses data.limit, which reads a prefix of the CSV and would restrict the "
            "ablation to the earliest capture days"
        )
        assert config.data.subsample_rows is not None


def test_open_set_holdout_is_viable():
    """Held-out apps must actually be unknown *in the test split*.

    An earlier version of configs/fiveg_open_set.yaml held out three apps of
    which two do not appear in the session-disjoint test split at all, so the
    evaluation would have scored AUROC against a single held-out app
    contributing 2.9% of test rows, all from one service class. The config
    would have run and produced a number; nothing would have flagged it.

    Skips when the dataset is absent, since CI does not carry it.
    """
    from pathlib import Path

    import pandas as pd
    import pytest as _pytest

    from flowconx.audit.splits import ensure_flow_ids, indices_from_manifest, load_manifest
    from flowconx.labels import RareClassPolicy, apply_rare_class_policy

    csv = Path("data/processed/fiveg_traffic.csv")
    manifest_path = Path("splits/fiveg_traffic/session_disjoint_seed42.json.gz")
    if not csv.exists() or not manifest_path.exists():
        _pytest.skip("5G dataset not built; run `python -m flowconx.data.prepare --source fiveg`")

    config = load_config("configs/fiveg_open_set.yaml")
    unknown = {a.lower() for a in config.data.unknown_apps}
    assert unknown, "the open-set config declares no held-out applications"

    frame = pd.read_csv(csv, usecols=["app", "service", "capture_id", "flow_id"])
    frame, _ = apply_rare_class_policy(
        frame, RareClassPolicy(mode=config.data.rare_class_mode, min_class_count=config.data.min_class_count)
    )
    frame, _ = ensure_flow_ids(frame)
    indices = indices_from_manifest(frame, load_manifest(manifest_path))
    train, test = frame.iloc[indices["train"]], frame.iloc[indices["test"]]

    present = unknown & set(test["app"].str.lower())
    assert present == unknown, f"held-out apps absent from the test split: {sorted(unknown - present)}"

    share = float(test["app"].str.lower().isin(unknown).mean())
    assert 0.05 < share < 0.5, f"unknowns are {share:.1%} of test; too lopsided to score AUROC on"

    services = set(frame[frame["app"].str.lower().isin(unknown)]["service"])
    assert len(services) >= 2, "held-out apps share one service; rejection collapses to rejecting a class"

    surviving = set(train[~train["app"].str.lower().isin(unknown)]["service"])
    assert surviving == set(frame["service"]), (
        "removing the held-out apps empties a service class from training, which makes the "
        "closed-set half of the open-set metric meaningless"
    )


def test_configs_differing_only_in_label_column_get_different_run_dirs():
    """A results path must encode everything that distinguishes an experiment.

    `results/baseline_<model>/<dataset>/<split>/seed<n>` did not encode the
    label column, so the application-task baselines wrote to the same path as
    the service-task ones and all twelve were skipped as "already present".
    Nothing reported it; the app-task comparison simply did not exist while
    appearing to.
    """
    service = load_config("configs/fiveg_main.yaml")
    app = load_config("configs/fiveg_app_task.yaml")
    assert service.data.label_column != app.data.label_column
    assert service.hash() != app.hash(), "different label columns must change the config hash"
    assert service.run_dir != app.run_dir, "different experiments must not share a results path"


def test_run_dir_distinguishes_every_axis_it_should():
    base = from_dict({"name": "x", "seed": 0})
    for override in (
        {"name": "y"},
        {"seed": 1},
        {"data": {"dataset": "other"}},
        {"data": {"split_protocol": "random_flow"}},
    ):
        assert from_dict({"name": "x", "seed": 0, **override}).run_dir != base.run_dir, override


def test_adversarial_weight_without_a_head_is_rejected():
    """A weight applied to a head that does not exist is a silent ablation.

    The adversarial head defaults to off, because a sweep over two orders of
    magnitude showed it removes nothing. That default made every
    `adv_weight_*` config inherit `adversarial_head: false` while still setting
    a non-zero weight -- configurations that would have run, produced numbers,
    and been tabulated as a weight sweep while sweeping nothing.
    """
    with pytest.raises(ValueError, match="adversarial_head=false"):
        from_dict({"model": {"adversarial_head": False}, "loss": {"lambda_adversarial": 0.5}})
    # Both consistent combinations are fine.
    from_dict({"model": {"adversarial_head": True}, "loss": {"lambda_adversarial": 0.5}}).validate()
    from_dict({"model": {"adversarial_head": False}, "loss": {"lambda_adversarial": 0.0}}).validate()


def test_adversarial_head_is_off_by_default():
    """The component is inert; nobody should inherit it by accident."""
    base = load_config("configs/base.yaml")
    assert base.model.adversarial_head is False
    assert base.loss.lambda_adversarial == 0.0


def test_the_sweep_configs_still_enable_the_head():
    from pathlib import Path

    for path in sorted(Path("configs/ablations").glob("adv_weight_*.yaml")):
        config = load_config(path)
        assert config.model.adversarial_head == (config.loss.lambda_adversarial > 0), path.name


def test_iscx_tor_vantage_inference_handles_the_traps():
    """Two mistakes the first implementation made, pinned so they cannot recur.

    `\\btor\\b` does not match `tor_skype` because `_` is a word character, and
    "nontor" contains "tor" so a NonTor capture whose filename mentions
    "gateway" must not be read as the Tor vantage.
    """
    from flowconx.data.iscx_tor import infer_vantage

    assert infer_vantage("Tor/AUDIO_tor_spotify2.pcap") == "gateway"
    assert infer_vantage("VOIP/tor_skype_audio.pcap") == "gateway"
    # Directory outranks a misleading filename token.
    assert infer_vantage("NonTor/browsing_gateway_chrome.pcap") == "workstation"
    assert infer_vantage("NonTor/AUDIO_spotify2.pcap") == "workstation"
    assert infer_vantage("workstation/email_imap.pcapng") == "workstation"
    # Says nothing rather than guessing.
    assert infer_vantage("misc/unlabelled_capture.pcap") == "unknown"


def test_iscx_tor_categories_do_not_shadow_each_other():
    from pathlib import Path

    from flowconx.data.iscx_tor import classify_path

    root = Path("/x")
    expected = {
        "Tor/audio_spotify.pcap": "audio_streaming",
        "Tor/video_youtube.pcap": "video_streaming",
        "NonTor/p2p_torrent.pcap": "p2p",
        "NonTor/email_imap.pcap": "email",
        "Tor/voip_skype.pcap": "voip",
        "NonTor/browsing_firefox.pcap": "browsing",
        "Tor/file_transfer_ftp.pcap": "file_transfer",
        "NonTor/chat_icq.pcap": "chat",
        # Capture files are numbered, and a trailing digit must not break the
        # token boundary: `spotify` has to match `spotify2`.
        "Tor/AUDIO_tor_spotify2.pcap": "audio_streaming",
        "NonTor/p2p_torrent01.pcap": "p2p",
        "NonTor/chat1a_icq.pcap": "chat",
        # An audio call is voip even though the app also does chat. The
        # specific variant must be tested before the bare app name.
        "Tor/hangouts_audio.pcap": "voip",
        "NonTor/hangouts_chat.pcap": "chat",
        "Tor/skype_audio2.pcap": "voip",
        "NonTor/skype_chat.pcap": "chat",
        # Says nothing rather than guessing.
        "misc/nothing.pcap": "unknown",
    }
    for path, service in expected.items():
        assert classify_path(root / path, root)[0] == service, path


def test_iscx_tor_loader_fails_loudly_without_data(tmp_path):
    """A missing corpus must raise with the reason, not return an empty list."""
    from flowconx.data.iscx_tor import IscxTorConfig, find_captures

    with pytest.raises(FileNotFoundError, match="registration form"):
        find_captures(IscxTorConfig(root=str(tmp_path / "absent")))
    (tmp_path / "present").mkdir()
    with pytest.raises(FileNotFoundError, match="ARFF"):
        find_captures(IscxTorConfig(root=str(tmp_path / "present")))
