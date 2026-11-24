import pandas as pd

from cxrseg.dataset import load_s3_params
from cxrseg.tune_clahe import pick_winner, update_s3_params


def test_winner_is_highest_and_ties_go_to_first():
    df = pd.DataFrame({"clip_limit": [1.0, 2.0, 3.0], "tile_grid": ["4x4", "8x8", "8x8"], "val_dice": [0.90, 0.95, 0.95]})
    w = pick_winner(df)
    assert w["clip_limit"] == 2.0 and w["tile_grid"] == "8x8"


def test_s3_params_roundtrip_and_replace(tmp_path):
    p = tmp_path / "s3_params.csv"
    update_s3_params(p, "ho_shenzhen", 3.0, "16x16")
    update_s3_params(p, "ho_darwin", 1.0, "4x4")
    out = update_s3_params(p, "ho_shenzhen", 4.0, "8x8")  # replaces, does not duplicate
    assert list(out["fold"]) == ["ho_darwin", "ho_shenzhen"]
    got = load_s3_params("ho_shenzhen", p)  # the format dataset.py reads
    assert got.clip_limit == 4.0 and got.tile_grid == (8, 8)
