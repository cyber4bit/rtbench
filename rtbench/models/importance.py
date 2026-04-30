from __future__ import annotations

from typing import Any

import numpy as np


def feature_group_importance(model: Any, group_sizes: dict[str, int]) -> dict[str, float]:
    if not hasattr(model, "feature_importances_"):
        return {group: 0.0 for group in group_sizes}
    imp = np.asarray(model.feature_importances_, dtype=np.float64)
    if imp.size == 0:
        return {group: 0.0 for group in group_sizes}
    out: dict[str, float] = {}
    start = 0
    for group, size in group_sizes.items():
        end = start + size
        out[group] = float(np.sum(imp[start:end]))
        start = end
    total = sum(out.values())
    if total > 0:
        out = {group: value / total for group, value in out.items()}
    return out
