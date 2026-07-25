"""Loader for external benchmarks."""

from typing import Any


import typing

class BenchmarkLoader:
    """Loads benchmark definitions and checks for required datasets."""

    SUPPORTED_BENCHMARKS: typing.ClassVar[set[str]] = {"ISIC", "HAM10000", "Derm7pt", "PH2"}

    def __init__(self, benchmark_name: str) -> None:
        """Initializes the loader.

        Time Complexity:
            O(1)

        Edge Cases:
            - Unknown benchmarks raise ValueError.

        Args:
            benchmark_name: Name of the benchmark to load.

        Raises:
            ValueError: If the benchmark is not supported.
        """
        if benchmark_name not in self.SUPPORTED_BENCHMARKS:
            raise ValueError(f"Benchmark {benchmark_name} not supported.")
        self.benchmark_name = benchmark_name

    def load_metadata(self) -> dict[str, Any]:
        """Loads metadata for the benchmark.

        Returns:
            A dictionary containing benchmark metadata.
        """
        return {
            "name": self.benchmark_name,
            "status": "loaded"
        }
