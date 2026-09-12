"""Parquet persistence for normalized market prices."""

from pathlib import Path

import pandas as pd


class ParquetPriceStorage:
    """Store normalized market prices in one Parquet file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        """Return whether the configured Parquet file exists."""
        return self.path.is_file()

    def load(self) -> pd.DataFrame:
        """Load normalized prices in deterministic order."""
        if not self.exists():
            raise FileNotFoundError(f"Price data file does not exist: {self.path}")

        frame = pd.read_parquet(self.path, engine="pyarrow")
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.sort_values(["date", "ticker"]).reset_index(drop=True)

    def save(self, frame: pd.DataFrame) -> None:
        """Save normalized prices in deterministic order."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stored = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
        stored.to_parquet(self.path, engine="pyarrow", index=False)
