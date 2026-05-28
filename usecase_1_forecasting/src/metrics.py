"""Evaluation metrics for the Store Sales forecasting task."""

from __future__ import annotations

import numpy as np


def rmsle(y_true, y_pred) -> float:
    """Compute root mean squared logarithmic error with non-negative predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.maximum(np.asarray(y_pred), 0)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))

