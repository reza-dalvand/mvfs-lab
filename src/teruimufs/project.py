import time
from typing import Optional

import numpy as np

from mvfs_core.paths import DEFAULT_DATASET
from mvfs_core.runner import ProjectRunResult

from .algorithm import TERUIMUFS
from .clustering import litekmeans
from .data_loader import load_dataset
from .metrics import EvaluationMetrics


class TERUIMUFSProject:
    name = "teruimufs"
    display_name = "TERUIMUFS"
    data_contract = "teruimufs-original"

    def run(
        self,
        data_path: Optional[str] = None,
        maxiter: int = 50,
        alpha: int = 10,
        beta: int = 1,
        tau: int = 100,
        pp: int = 1,
    ) -> ProjectRunResult:
        if data_path is None:
            data_path = DEFAULT_DATASET

        dataset_name = str(data_path)

        (
            X,
            labels,
            missing_num,
            one_indices,
            XF,
            W,
            instance_num,
            class_num,
        ) = load_dataset(str(data_path))

        view_num = len(X)
        omega = np.ones(view_num)

        print("TERUIMUFS...")
        t0 = time.time()

        ranking = TERUIMUFS(
            X,
            XF,
            W,
            missing_num,
            class_num,
            maxiter,
            alpha,
            beta,
            tau,
            omega,
            pp,
        )

        runtime = time.time() - t0
        print(f"TERUIMUFS completed in {runtime:.2f} seconds.")

        XFT = XF.T
        dim_num = XF.shape[0]

        metrics_by_ratio = []

        print("در حال کلاسترینگ و ارزیابی...")

        for t1 in range(9):
            prop = (t1 + 2) * 0.05
            k = max(1, int(np.floor(prop * dim_num)))
            Xsub = XFT[:, ranking[:k]]

            pred = litekmeans(Xsub, class_num, Replicates=20)
            R = EvaluationMetrics(labels, pred)

            row = {
                "prop": prop,
                "acc": R[0],
                "nmi": R[1],
                "purity": R[2],
                "fscore": R[3],
                "precision": R[4],
                "recall": R[5],
                "ari": R[6],
            }
            metrics_by_ratio.append(row)

            print(
                f"Prop {prop:.2f}: "
                f"ACC={row['acc']:.4f}, "
                f"NMI={row['nmi']:.4f}, "
                f"Purity={row['purity']:.4f}"
            )

        best_keys = ["acc", "nmi", "purity", "fscore", "ari"]
        best_metrics = {
            key: max(row[key] for row in metrics_by_ratio)
            for key in best_keys
        }

        print("نتایج نهایی (Missing Ratio 0.2):")
        print(
            f"ACC: {best_metrics['acc'] * 100:.2f}%, "
            f"NMI: {best_metrics['nmi'] * 100:.2f}%, "
            f"Purity: {best_metrics['purity'] * 100:.2f}%, "
            f"Fscore: {best_metrics['fscore'] * 100:.2f}%, "
            f"ARI: {best_metrics['ari'] * 100:.2f}%"
        )

        config = {
            "data_path": str(data_path),
            "maxiter": maxiter,
            "alpha": alpha,
            "beta": beta,
            "tau": tau,
            "pp": pp,
        }

        extra = {
            "labels": labels,
            "missing_num": missing_num,
            "one_indices": one_indices,
            "XF": XF,
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