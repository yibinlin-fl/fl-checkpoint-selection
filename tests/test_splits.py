import numpy as np
import pytest
from p1.splits import make_splits, validate_splits


def cfg(partition="iid"):
    return {"seed": 123, "num_clients": 4, "partition": partition,
            "dirichlet_alpha": 0.7, "validation_fraction": 0.2}


@pytest.mark.parametrize("partition", ["iid", "dirichlet"])
def test_split_disjoint_and_full(partition):
    labels = np.repeat(np.arange(10), 80)
    tr, val = make_splits(labels, cfg(partition))
    validate_splits(tr, val, len(labels))
    tr2, val2 = make_splits(labels, cfg(partition))
    assert tr == tr2 and val == val2


def test_duplicate_or_intersect_rejected():
    with pytest.raises(AssertionError):
        validate_splits({0:[0,1]}, {0:[1,2]}, 4)
