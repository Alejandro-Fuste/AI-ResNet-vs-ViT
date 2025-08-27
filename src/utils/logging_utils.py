# src/utils/logging_utils.py
import csv
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Optional


def create_logger(name: str = "cvsys", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create a simple logger that logs to stdout and (optionally) to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class CSVLogger:
    """
    Lightweight CSV logger for per-epoch metrics.
    Use:
        csvlog = CSVLogger('logs/resnet18_train.csv', fieldnames=['epoch','train_loss','val_loss','train_acc','val_acc','lr','secs'])
        csvlog.log({'epoch':1, 'train_loss':..., ...})
    """
    def __init__(self, path: str, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)

        # Write header if file doesn't exist
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict):
        # Keep only known fields; missing fields become empty strings
        row_clean = {k: row.get(k, "") for k in self.fieldnames}
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row_clean)


class Timer:
    """
    Context manager for timing code blocks.

    Example:
        with Timer() as t:
            train_one_epoch(...)
        secs = t.elapsed
    """
    def __enter__(self):
        self._t0 = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = perf_counter() - self._t0
        return False  # don't suppress exceptions
