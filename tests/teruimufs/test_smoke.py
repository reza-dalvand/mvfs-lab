import numpy as np

from teruimufs.algorithm import TERUIMUFS


def test_teruimufs_smoke_direct_algorithm():
    """
    Smoke test for TERUIMUFS using a tiny synthetic multi-view dataset.

    This test directly calls the original algorithm to avoid depending
    on an external .mat file.
    """
    np.random.seed(0)

    n = 40
    class_num = 2
    dims = [5, 6]
    missing_num = [3, 4]

    X = [np.random.randn(d, n) for d in dims]
    XF = np.vstack(X)

    W = []
    all_indices = np.arange(n)

    for m in missing_num:
        missing_idx = np.random.choice(all_indices, size=m, replace=False)
        observed_idx = np.setdiff1d(all_indices, missing_idx)

        Wv = np.eye(n)
        Wv = np.delete(Wv, observed_idx, axis=0)
        W.append(Wv)

    omega = np.ones(len(X))

    ranking = TERUIMUFS(
        X,
        XF,
        W,
        missing_num,
        class_num,
        maxiter=1,
        alpha=10,
        beta=1,
        tau=100,
        omega=omega,
        pp=1,
    )

    total_dim = sum(dims)

    assert ranking.ndim == 1
    assert len(ranking) == total_dim
    assert len(np.unique(ranking)) == total_dim