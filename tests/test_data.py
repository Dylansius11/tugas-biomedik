import pandas as pd
import pytest

from cxrseg.data import FOLDS, fold_split, held_out_source


def make_manifest():
    rows = []
    for source, n in (("darwin", 50), ("montgomery", 20), ("shenzhen", 30)):
        for i in range(n):
            split = "train" if i < 0.8 * n else ("val" if i < 0.9 * n else "test")
            rows.append({"id": f"{source}/{i}", "source": source, "split": split})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("fold", FOLDS)
def test_held_out_source_never_leaks_into_fitting_sets(fold):
    m = make_manifest()
    ho = held_out_source(fold)
    s = fold_split(m, fold)
    for part in ("train", "val", "in_source_test"):
        assert ho not in set(s[part]["source"])
    assert set(s["foreign"]["source"]) == {ho}
    assert len(s["foreign"]) == (m["source"] == ho).sum()


def test_parts_are_disjoint_and_cover_everything():
    m = make_manifest()
    s = fold_split(m, "ho_darwin")
    ids = [set(s[k]["id"]) for k in ("train", "val", "in_source_test", "foreign")]
    assert sum(len(x) for x in ids) == len(m)
    assert len(set.union(*ids)) == len(m)


def test_unknown_fold_is_rejected():
    with pytest.raises(ValueError):
        held_out_source("ho_unknown")
