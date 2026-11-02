from .runner import ProjectRunResult


def print_result_summary(result: ProjectRunResult) -> None:
    print()
    print("=" * 80)
    print(f"Project: {result.project}")
    print(f"Dataset: {result.dataset}")
    print(f"Runtime: {result.runtime_seconds:.2f} seconds")
    print(f"Ranking size: {len(result.ranking)}")
    print("-" * 80)

    for row in result.metrics_by_ratio:
        line = f"Prop {row['prop']:.2f}:"
        for key, value in row.items():
            if key == "prop":
                continue
            line += f" {key}={value:.4f}"
        print(line)

    if result.best_metrics:
        best_line = "Best: "
        best_line += ", ".join(
            f"{key}={value * 100:.2f}%"
            for key, value in result.best_metrics.items()
        )
        print(best_line)

    print("=" * 80)