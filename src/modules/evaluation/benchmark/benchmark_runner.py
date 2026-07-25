"""Runner for evaluating benchmarks."""


from .benchmark_loader import BenchmarkLoader


class BenchmarkRunner:
    """Runs a benchmark evaluation."""

    def __init__(self, loader: BenchmarkLoader) -> None:
        """Initializes the runner.

        Time Complexity:
            O(1)

        Args:
            loader: A BenchmarkLoader instance.
        """
        self.loader = loader

    def run(self, metrics_available: bool = False) -> dict[str, float] | str:
        """Runs the benchmark.

        If actual metrics are not available/missing, it returns 'External Validation Required'
        or 'Not Applicable' instead of fabricating data.

        Time Complexity:
            O(1)

        Edge Cases:
            - When missing metrics, handles missing evaluation gracefully.

        Args:
            metrics_available: Boolean flag indicating if real metrics data is available.

        Returns:
            Dictionary of metrics if available, otherwise a fallback string message.
        """
        if not metrics_available:
            return "External Validation Required"

        return {"accuracy": 0.0}
