from uimufslr.project import UIMUFSLRProject


def test_uimufslr_smoke_with_synthetic_fallback():
    adapter = UIMUFSLRProject()

    synthetic_options = {
        "n_samples": 60,
        "view_dims": [8, 9, 10],
        "n_clusters": 2,
        "missing_per": 20,
    }

    result = adapter.run(
        data_path="nonexistent_dataset.mat",
        max_iter=1,
        neighbor_num=3,
        seed=0,
        fallback_synthetic=True,
        synthetic_options=synthetic_options,
    )

    assert result.project == "uimufslr"
    assert result.dataset == "synthetic-uimufslr"
    assert result.ranking.ndim == 1
    assert len(result.ranking) == sum(synthetic_options["view_dims"])
    assert len(result.metrics_by_ratio) == 9

    for key in ["acc", "nmi", "purity", "fscore", "precision"]:
        assert key in result.best_metrics
        assert 0.0 <= result.best_metrics[key] <= 1.0