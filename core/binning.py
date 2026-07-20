"""学習データでfitしたqcut境界を使う汎用ビン留め。

固定の数値境界（旧 core/bins.py）を人が勘で決めるのではなく、学習期間のデータ分布から
qcutで境界を決め、その数値境界を保存して以後（検証期間・当日データ）に再利用する。
境界は学習データの範囲外の値も同じ端のビンに入るよう、両端を±infに拡張して保存する
（旧bins.pyの -999/999 という番兵と同じ考え方）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_qcut_edges(train_values, q: int) -> np.ndarray:
    """学習データからqcutの境界値(q+1個、両端は±inf)を求める。有効値が無ければ空配列。"""
    v = pd.to_numeric(pd.Series(train_values), errors="coerce").dropna()
    if v.empty:
        return np.array([])
    try:
        _, edges = pd.qcut(v, q=q, duplicates="drop", retbins=True)
    except ValueError:
        return np.array([])
    edges = np.asarray(edges, dtype=float).copy()
    if len(edges) < 2:
        return np.array([])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_edges(values, edges: np.ndarray, label_prefix: str = "bin") -> pd.Series:
    """保存済みの境界値(edges)を使って値をビン留めする（pd.cut）。"""
    idx = values.index if hasattr(values, "index") else None
    if edges is None or len(edges) < 2:
        return pd.Series(np.nan, index=idx)
    v = pd.to_numeric(pd.Series(values), errors="coerce")
    labels = [f"{label_prefix}{i + 1}" for i in range(len(edges) - 1)]
    return pd.cut(v, bins=edges, labels=labels, include_lowest=True)


def edges_to_bounds_df(edges: np.ndarray, label_prefix: str = "bin") -> pd.DataFrame:
    """ビンラベルごとの (label, low, high, include_lowest) 対応表。ルールCSVの保存・当日判定に使う。"""
    if edges is None or len(edges) < 2:
        return pd.DataFrame(columns=["label", "low", "high", "include_lowest"])
    labels = [f"{label_prefix}{i + 1}" for i in range(len(edges) - 1)]
    rows = [
        {"label": lab, "low": float(edges[i]), "high": float(edges[i + 1]), "include_lowest": i == 0}
        for i, lab in enumerate(labels)
    ]
    return pd.DataFrame(rows)


def in_interval(val, low, high, include_lowest) -> bool:
    if pd.isna(val) or pd.isna(low) or pd.isna(high):
        return False
    if bool(include_lowest):
        return (val >= low) and (val <= high)
    return (val > low) and (val <= high)
