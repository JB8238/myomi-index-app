import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import re
import io
from datetime import datetime


# =========================================================
# 分析表示（回収率/的中率）
# =========================================================
def calc_roi_table(src: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    回収率は「外れ=0円」で平均（単勝/複勝は100円あたりの払戻円を想定）
    的中率は bool の平均（%表示用に×100）
    """
    d = src.copy()
    # 外れ（NaN）を 0 として回収率計算
    d["単勝_回収"] = pd.to_numeric(d.get("単勝"), errors="coerce").fillna(0)
    d["複勝_回収"] = pd.to_numeric(d.get("複勝"), errors="coerce").fillna(0)

    out = (
        d.groupby(group_col, observed=True)
        .agg(
            頭数=("馬名", "count") if "馬名" in d.columns else ("馬番", "count"),
            単勝的中率=("単勝的中", "mean"),
            複勝的中率=("複勝的中", "mean"),
            単勝回収率=("単勝_回収", "mean"),
            複勝回収率=("複勝_回収", "mean"),
        )
        .reset_index()
    )

    # 表示用（%）
    out["単勝的中率"] = out["単勝的中率"] * 100
    out["複勝的中率"] = out["複勝的中率"] * 100
    # 単勝回収率/複勝回収率は「円/100円」なので数値=％相当（例: 95.2 → 95.2%）

    return out

# =========================================================
# ヒートマップによる集計
# =========================================================
def make_heatmap_table(d: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    value_col:
        - '単勝ROI' / '複勝ROI' / '単勝的中率' / '複勝的中率' / '件数'
    """
    x = d.copy()

    # 区分列が無い時は落ちないように
    if "上昇値区分" not in x.columns or "人気乖離区分" not in x.columns:
        return pd.DataFrame()

    # ROI用に「外れ=0」を作る（NaNの平均を避ける）
    x["単勝_回収"] = pd.to_numeric(x.get("単勝"), errors="coerce").fillna(0)
    x["複勝_回収"] = pd.to_numeric(x.get("複勝"), errors="coerce").fillna(0)

    pivot_count = x.pivot_table(
        index="上昇値区分", columns="人気乖離区分",
        values="馬番" if "馬番" in x.columns else "馬名",
        aggfunc="count", fill_value=0, observed=True
    )

    pivot_win_roi = x.pivot_table(
        index="上昇値区分", columns="人気乖離区分",
        values="単勝_回収", aggfunc="mean", observed=True
    )

    pivot_plc_roi = x.pivot_table(
        index="上昇値区分", columns="人気乖離区分",
        values="複勝_回収", aggfunc="mean", observed=True
    )

    pivot_win_hit = x.pivot_table(
        index="上昇値区分", columns="人気乖離区分",
        values="単勝的中", aggfunc="mean", observed=True
    ) * 100

    pivot_plc_hit = x.pivot_table(
        index="上昇値区分", columns="人気乖離区分",
        values="複勝的中", aggfunc="mean", observed=True
    ) * 100

    if value_col == "件数":
        return pivot_count
    if value_col == "単勝ROI":
        return pivot_win_roi
    if value_col == "複勝ROI":
        return pivot_plc_roi
    if value_col == "単勝的中率":
        return pivot_win_hit
    if value_col == "複勝的中率":
        return pivot_plc_hit

    return pd.DataFrame()

# =========================================================
# 買い条件自動抽出
# =========================================================
def build_condition_cells(df_hm: pd.DataFrame) -> pd.DataFrame:
    x = df_hm.copy()

    # ROI計算用（はずれ=0）
    x["単勝_回収"] = pd.to_numeric(x.get("単勝"), errors="coerce").fillna(0)
    x["複勝_回収"] = pd.to_numeric(x.get("複勝"), errors="coerce").fillna(0)

    agg = (
        x.groupby(
            ["レースレベル", "上昇値区分", "人気乖離区分"],
            observed=True
        )
        .agg(
            件数=("馬番", "count"),
            単勝ROI=("単勝_回収", "mean"),
            単勝的中率=("単勝的中", "mean"),
            複勝ROI=("複勝_回収", "mean"),
            複勝的中率=("複勝的中", "mean"),
        )
        .reset_index()
    )

    agg["単勝的中率"] = agg["単勝的中率"] * 100
    agg["複勝的中率"] = agg["複勝的中率"] * 100
    return agg

def extract_buy_conditions_win(
    cells: pd.DataFrame,
    min_roi: float = 110.0,
    min_n: int = 10
) -> pd.DataFrame:
    out = cells[
        (cells["単勝ROI"] >= min_roi) &
        (cells["件数"] >= min_n)
    ].copy()

    # Lv → 上昇値 → 人気乖離 → ROI の順で見やすく
    out = out.sort_values(
        ["レースレベル", "単勝ROI", "件数"],
        ascending=[True, False, False]
    )

    return out

def extract_buy_conditions_place(
    cells: pd.DataFrame,
    min_roi: float = 105.0,
    min_n: int = 10
) -> pd.DataFrame:
    out = cells[
        (cells["複勝ROI"] >= min_roi) &
        (cells["件数"] >= min_n)
    ].copy()

    # Lv → 上昇値 → 人気乖離 → ROI の順で見やすく
    out = out.sort_values(
        ["レースレベル", "複勝ROI", "件数"],
        ascending=[True, False, False]
    )

    return out