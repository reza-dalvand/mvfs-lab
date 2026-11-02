import time
from typing import Any, Dict, Optional

import numpy as np

from mvfs_core.paths import DEFAULT_DATASET
from mvfs_core.runner import ProjectRunResult

from .algorithm import uimufslr
from .clustering import evaluate_feature_selection
from .data_loader import load_dataset
from .preprocessing import generate_synthetic_dataset


class UIMUFSLRProject:
    name = "uimufslr"
    display_name = "UIMUFSLR"
    data_contract = "uimufslr-original"

    def run(
        self,
        data_path: Optional[str] = None,
        max_iter: int = 50,
        neighbor_num: int = 5,
        alpha: float = 10.0,
        beta: float = 1.0,
        tau: float = 1.0,
        lambda_: float = 1.0,
        seed: int = 42,
        fallback_synthetic: bool = True,
        synthetic_options: Optional[Dict[str, Any]] = None,
    ) -> ProjectRunResult:
        if data_path is None:
            data_path = DEFAULT_DATASET

        np.random.seed(seed)

        dataset_name = str(data_path)

        try:
            X_missing, XF, M, zero_indices, labels, class_num, n = load_dataset(str(data_path))
        except FileNotFoundError:
            if not fallback_synthetic:
                raise

            dataset_name = "synthetic-uimufslr"
            default_synthetic_options = dict(
                n_samples=300,
                view_dims=[50, 60, 70],
                n_clusters=5,
                missing_per=20,
                random_state=seed,
            )
            if synthetic_options:
                default_synthetic_options.update(synthetic_options)

            X_missing, XF, M, zero_indices, labels, class_num, n = generate_synthetic_dataset(
                **default_synthetic_options
            )

        print(f"Dataset loaded: {len(X_missing)} views, {n} samples, {class_num} classes.")

        print("Running UIMUFSLR...")
        t0 = time.time()

        ranking, C, Y = uimufslr(
            X_missing,
            XF,
            M,
            zero_indices,
            class_num,
            maxiter=max_iter,
            neighbor_num=neighbor_num,
            alpha=alpha,
            beta=beta,
            tau=tau,
            lambda_=lambda_,
            random_state=seed,
        )

        runtime = time.time() - t0
        print(f"UIMUFSLR completed in {runtime:.2f} seconds.")

        print("Evaluating feature selection...")
        t1 = time.time()

        dim_num = XF.shape[0]
        XFT = XF.T

        metrics_by_ratio = []

        for t1_idx in range(1, 10):
            prop = (t1_idx + 1) * 0.05
            k = max(1, int(np.floor(prop * dim_num)))
            selected_features = ranking[:k]
            Xsub = XFT[:, selected_features]

            row = evaluate_feature_selection(
                Xsub,
                labels,
                class_num,
                random_state=seed,
            )
            row = {"prop": prop, **row}
            metrics_by_ratio.append(row)

            print(
                f"Prop {prop:.2f}: "
                f"ACC={row['acc']:.4f}, "
                f"NMI={row['nmi']:.4f}, "
                f"Purity={row['purity']:.4f}"
            )

        print(f"Evaluation completed in {time.time() - t1:.2f} seconds.")

        best_keys = ["acc", "nmi", "purity", "fscore", "precision"]
        best_metrics = {
            key: max(row[key] for row in metrics_by_ratio)
            for key in best_keys
        }

        print(
            "result with the optimal feature selection percentage while missing ratio 0.2: "
            f"ACC {best_metrics['acc'] * 100:.2f}%, "
            f"NMI {best_metrics['nmi'] * 100:.2f}%, "
            f"Purity {best_metrics['purity'] * 100:.2f}%, "
            f"Fscore {best_metrics['fscore'] * 100:.2f}%, "
            f"Precision {best_metrics['precision'] * 100:.2f}%"
        )

        config = {
            "data_path": str(data_path),
            "max_iter": max_iter,
            "neighbor_num": neighbor_num,
            "alpha": alpha,
            "beta": beta,
            "tau": tau,
            "lambda_": lambda_,
            "seed": seed,
            "fallback_synthetic": fallback_synthetic,
        }

        extra = {
            "C": C,
            "Y": Y,
            "labels": labels,
            "XF": XF,
            "zero_indices": zero_indices,
        }

        return ProjectRunResult(
            project=self.name,
            dataset=dataset_name,
            ranking=ranking,
            metrics_by_ratio=metrics_by_ratio,
            best_metrics=best_metrics,
            runtime_seconds=runtime,
            config=config,
            extra=extra,
        )