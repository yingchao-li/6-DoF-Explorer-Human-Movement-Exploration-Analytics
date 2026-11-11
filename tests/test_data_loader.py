"""Tests for CSV import, validation and source separation."""
import io

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_dataset
from src.data_loader import ColumnMapping, ImportSettings, guess_mapping, load_csv, parse_table
from src.features import euler_zyx_to_quat


def raw_table(n=120, fs=60.0, with_quat=True, unit_scale=100.0, t_scale=1000.0):
    t = np.arange(n) / fs
    d = {"subj": ["s1"] * n, "grp": ["ctl"] * n, "t_ms": t * t_scale,
         "px": t * unit_scale, "py": np.zeros(n), "pz": np.full(n, 1.6 * unit_scale)}
    if with_quat:
        q = euler_zyx_to_quat(0.5 * t, np.zeros(n), np.zeros(n))
        # deliberately WXYZ order, to prove per-component mapping removes order ambiguity
        d.update({"qw_": q[:, 3], "qx_": q[:, 0], "qy_": q[:, 1], "qz_": q[:, 2]})
    return pd.DataFrame(d)


MAP = ColumnMapping(participant="subj", timestamp="t_ms", x="px", y="py", z="pz",
                    qx="qx_", qy="qy_", qz="qz_", qw="qw_", condition="grp")


def test_valid_upload_with_unit_conversion_and_component_mapping():
    ds, rep = parse_table(raw_table(), MAP, ImportSettings(time_unit="ms", position_unit="cm"))
    assert rep.ok and ds is not None
    assert ds.source == "uploaded" and not ds.is_synthetic
    assert ds.has_rotation and ds.has_condition and ds.unit_label == "m"
    assert ds.df.timestamp.iloc[0] == 0.0
    assert ds.df.timestamp.iloc[1] == pytest.approx(1 / 60)
    assert ds.df.position_x.iloc[-1] == pytest.approx(119 / 60)       # cm -> m
    assert rep.info["median_rate_hz"] == pytest.approx(60.0)
    np.testing.assert_allclose(ds.df.quaternion_w.iloc[:3], np.cos(0.25 * np.arange(3) / 60), atol=1e-9)


def test_missing_quaternions_disable_rotation_and_condition_optional():
    m = ColumnMapping(participant=None, timestamp="t_ms", x="px", y="py", z="pz")
    ds, rep = parse_table(raw_table(with_quat=False), m, ImportSettings(time_unit="ms"))
    assert rep.ok and not ds.has_rotation and not ds.has_condition
    assert ds.df.participant_id.unique().tolist() == ["P01"] and ds.df.condition.unique().tolist() == ["All"]
    assert any("rotation" in w for w in rep.warnings)
    assert ds.df[["quaternion_x", "quaternion_w"]].isna().all().all()


@pytest.mark.parametrize("changes,msg", [
    (dict(qw=None), "incomplete"),
    (dict(x="nope"), "not selected or not found"),
    (dict(y="px"), "mapped twice"),
])
def test_invalid_mappings_are_rejected(changes, msg):
    m = ColumnMapping(**{**MAP.__dict__, **changes})
    ds, rep = parse_table(raw_table(), m, ImportSettings(time_unit="ms"))
    assert ds is None and any(msg in e for e in rep.errors)


def test_non_numeric_and_unusable_columns_are_reported():
    df = raw_table()
    df["px"] = df["px"].astype(object)
    df.loc[3, "px"] = "oops"
    ds, rep = parse_table(df, MAP, ImportSettings(time_unit="ms"))
    assert rep.ok and any("non-numeric" in w for w in rep.warnings)
    assert ds.df.position_x.isna().sum() == 1                          # converted to missing, not dropped
    df["py"] = "text"
    ds, rep = parse_table(df, MAP, ImportSettings(time_unit="ms"))
    assert ds is None and any("no numeric" in e for e in rep.errors)


def test_wrong_time_unit_is_flagged_by_the_rate_check():
    ds, rep = parse_table(raw_table(), MAP, ImportSettings(time_unit="s"))      # ms interpreted as s
    assert any("sampling rate" in w for w in rep.warnings)


def test_datetime_timestamps():
    df = raw_table()
    df["t_ms"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(len(df)) / 60.0, unit="s")
    ds, rep = parse_table(df, MAP, ImportSettings(time_unit="datetime", position_unit="cm"))
    assert rep.ok and ds.df.timestamp.iloc[-1] == pytest.approx(119 / 60, abs=1e-6)


def test_missing_participant_ids_are_errors_not_silently_dropped():
    df = raw_table()
    df.loc[2, "subj"] = None
    ds, rep = parse_table(df, MAP, ImportSettings(time_unit="ms"))
    assert ds is None and any("participant" in e for e in rep.errors)


def test_non_unit_quaternions_are_warned_about():
    df = raw_table()
    for c in ("qw_", "qx_", "qy_", "qz_"):
        df[c] *= 5
    ds, rep = parse_table(df, MAP, ImportSettings(time_unit="ms"))
    assert rep.ok and any("norm" in w for w in rep.warnings)


def test_file_order_is_preserved_so_preprocessing_can_report_unsorted_input():
    df = raw_table().iloc[::-1].reset_index(drop=True)
    ds, _ = parse_table(df, MAP, ImportSettings(time_unit="ms"))
    assert ds.df.timestamp.is_monotonic_decreasing


def test_csv_roundtrip_with_semicolon_delimiter_and_bad_file():
    text = raw_table().to_csv(index=False, sep=";").encode()
    ds, rep = load_csv(io.BytesIO(text), MAP, ImportSettings(time_unit="ms", position_unit="cm"))
    assert rep.ok and len(ds.df) == 120
    ds, rep = load_csv(io.BytesIO(b"\x00\x01\x02"), MAP, ImportSettings())
    assert ds is None and rep.errors


def test_guess_mapping():
    g = guess_mapping(["participant_id", "timestamp", "position_x", "position_y", "position_z",
                       "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w", "condition"])
    assert g["participant"] == "participant_id" and g["x"] == "position_x" and g["qw"] == "quaternion_w"
    assert g["condition"] == "condition" and g["timestamp"] == "timestamp"


def test_synthetic_and_uploaded_datasets_cannot_be_confused():
    syn = generate_dataset(n_participants=4, seed=1)
    up, _ = parse_table(raw_table(), MAP, ImportSettings(time_unit="ms", position_unit="cm"))
    assert syn.dataset_id.startswith("synthetic:") and up.dataset_id.startswith("uploaded:")
    assert syn.dataset_id != up.dataset_id
    # uploading the synthetic CSV itself still yields an *uploaded* dataset with its own id
    again, _ = parse_table(syn.df, ColumnMapping("participant_id", "timestamp", "position_x", "position_y",
                           "position_z", "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w",
                           "condition"), ImportSettings())
    assert again.dataset_id.startswith("uploaded:")


def test_guess_mapping_recognises_common_column_names():
    from src.data_loader import guess_mapping
    g = guess_mapping(["sub", "grp", "time_ms", "X", "Y", "Z", "qx", "qy", "qz", "qw"])
    assert (g["participant"], g["condition"], g["timestamp"]) == ("sub", "grp", "time_ms")
    assert (g["x"], g["y"], g["z"]) == ("X", "Y", "Z")
    assert (g["qx"], g["qy"], g["qz"], g["qw"]) == ("qx", "qy", "qz", "qw")
    h = guess_mapping(["participant_id", "condition", "timestamp", "position_x", "position_y", "position_z",
                       "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w", "euler_yaw_deg"])
    assert h["participant"] == "participant_id" and h["qw"] == "quaternion_w" and h["x"] == "position_x"
