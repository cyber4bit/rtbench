from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def _slug(text: str) -> str:
    text = str(text or "").strip().lower()
    if not text:
        return "__missing__"
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text if text else "__missing__"


def _split_words(text: str) -> list[str]:
    raw = str(text or "").strip().lower()
    if not raw:
        return ["__missing__"]
    parts = re.split(r"[^a-z0-9]+", raw)
    toks = [_slug(p) for p in parts if p]
    return toks if toks else ["__missing__"]


def _as_float(x: Any) -> float:
    try:
        return float(pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0])
    except Exception:
        return float("nan")


def _unit_scale(unit: str) -> float:
    u = str(unit or "").strip()
    if u == "碌M":
        return 1.0 / 1000.0
    return 1.0


def _gradient_segments(gradient_df: pd.DataFrame) -> list[np.ndarray]:
    if gradient_df.empty or gradient_df.shape[0] < 2:
        return []
    time_col = gradient_df.columns[0]
    cols = [c for c in gradient_df.columns if any(k in c.lower() for k in ("a [", "b [", "c [", "d [", "flow"))]
    if not cols:
        return []
    df = gradient_df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.loc[df[time_col].notna()].sort_values(time_col).reset_index(drop=True)
    if df.shape[0] < 2:
        return []
    segs: list[np.ndarray] = []
    for i in range(df.shape[0] - 1):
        t0 = float(df.loc[i, time_col])
        t1 = float(df.loc[i + 1, time_col])
        dt = t1 - t0
        if not np.isfinite(dt) or dt <= 0:
            continue
        v0 = pd.to_numeric(df.loc[i, cols], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        v1 = pd.to_numeric(df.loc[i + 1, cols], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if v0.size < 5:
            v0 = np.pad(v0, (0, 5 - v0.size), mode="constant")
            v1 = np.pad(v1, (0, 5 - v1.size), mode="constant")
        if v0.size > 5:
            v0 = v0[:5]
            v1 = v1[:5]
        seg = np.concatenate([np.array([dt], dtype=np.float32), v0.astype(np.float32), v1.astype(np.float32)], axis=0)
        segs.append(seg.astype(np.float32))
    return segs


__all__ = ["_as_float", "_gradient_segments", "_slug", "_split_words", "_unit_scale"]
