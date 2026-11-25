import cv2
import numpy as np
import pandas as pd

from cxrseg.config import load_config
from cxrseg.evaluate import COLUMNS, evaluate_all, evaluate_run


def _make_synthetic_processed(tmp_path):
    proc = tmp_path / "processed"
    rows = []
    for source in ("darwin", "montgomery", "shenzhen"):
        (proc / source / "images").mkdir(parents=True, exist_ok=True)
        (proc / source / "masks").mkdir(parents=True, exist_ok=True)
        for i, split in enumerate(("train", "val", "test")):
            img = np.full((256, 256), 80 + i * 10, dtype=np.uint8)
            msk = np.zeros((256, 256), dtype=np.uint8)
            msk[60:180, 40:100] = 255
            msk[60:180, 150:210] = 255
            name = f"img_{i}"
            img_rel = f"{source}/images/{name}.png"
            msk_rel = f"{source}/masks/{name}.png"
            cv2.imwrite(str(proc / img_rel), img)
            cv2.imwrite(str(proc / msk_rel), msk)
            rows.append(
                {
                    "id": f"{source}/{name}",
                    "source": source,
                    "name": name,
                    "image_path": img_rel,
                    "mask_path": msk_rel,
                    "orig_h": 512,
                    "orig_w": 512,
                    "mask_area_frac": float((msk > 0).mean()),
                    "split": split,
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(proc / "manifest.csv", index=False)
    return proc, manifest


def test_perfect_and_empty_predictions_and_row_count(tmp_path):
    cfg = load_config()
    proc, manifest = _make_synthetic_processed(tmp_path)

    # Load ground truth masks for 3 images (val, in_source, foreign)
    sub = manifest[manifest["id"].isin(["montgomery/img_1", "shenzhen/img_2", "darwin/img_0"])].copy()
    masks = [cv2.imread(str(proc / r.mask_path), cv2.IMREAD_GRAYSCALE) for r in sub.itertuples()]

    # Case 1: Perfect predictions (prob_u8 == mask 0/255)
    pred_dir = tmp_path / "preds" / "ho_darwin" / "S0" / "seed11"
    pred_dir.mkdir(parents=True)
    perfect_npz = pred_dir / "pred.npz"
    np.savez_compressed(
        perfect_npz,
        ids=np.array(sub["id"].tolist()),
        group=np.array(["val", "in_source", "foreign"]),
        prob_u8=np.stack(masks),
    )

    df_perf = evaluate_run(perfect_npz, manifest, proc, cfg)
    assert list(df_perf.columns) == COLUMNS
    assert len(df_perf) == len(sub)
    assert np.allclose(df_perf["dice"], 1.0)
    assert (~df_perf["failed"]).all()
    assert (df_perf["fold"] == "ho_darwin").all()
    assert (df_perf["scenario"] == "S0").all()
    assert (df_perf["seed"] == 11).all()

    # Case 2: Empty predictions (all zeros)
    empty_npz = tmp_path / "preds" / "ho_darwin" / "S1" / "seed22" / "pred.npz"
    empty_npz.parent.mkdir(parents=True)
    np.savez_compressed(
        empty_npz,
        ids=np.array(sub["id"].tolist()),
        group=np.array(["val", "in_source", "foreign"]),
        prob_u8=np.zeros_like(np.stack(masks)),
    )

    df_empty = evaluate_run(empty_npz, manifest, proc, cfg)
    assert len(df_empty) == len(sub)
    assert np.allclose(df_empty["hd95"], cfg["metrics"]["hd95_empty_value"])
    assert df_empty["failed"].all()


def test_evaluate_all_writes_expected_csv_structure(tmp_path):
    cfg = load_config()
    proc, manifest = _make_synthetic_processed(tmp_path)
    sub = manifest[manifest["id"].isin(["montgomery/img_1", "darwin/img_0"])].copy()
    masks = [cv2.imread(str(proc / r.mask_path), cv2.IMREAD_GRAYSCALE) for r in sub.itertuples()]

    npz_path = tmp_path / "preds" / "ho_darwin" / "S0" / "seed11" / "pred.npz"
    npz_path.parent.mkdir(parents=True)
    np.savez_compressed(
        npz_path,
        ids=np.array(sub["id"].tolist()),
        group=np.array(["val", "foreign"]),
        prob_u8=np.stack(masks),
    )

    out_dir = tmp_path / "metrics"
    df = evaluate_all(tmp_path / "preds", out_dir, cfg, proc)
    expected_csv = out_dir / "ho_darwin" / "S0" / "seed11.csv"
    assert expected_csv.exists()
    loaded = pd.read_csv(expected_csv)
    assert len(loaded) == len(df) == 2
    assert list(loaded.columns) == COLUMNS
