import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import re
import io
from datetime import datetime


def add_component_pass_count(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    総合利益度>=0の馬について、
    騎手/調教師/種牡馬 利益度が >=0 のカテゴリ数を数える
    """
    d = df_in.copy()

    def _ok(x):
        return pd.notna(x) and x >= 0

    cnt = 0
    for col in ["騎手利益度", "調教師利益度", "種牡馬利益度"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
            cnt += d[col].apply(_ok).astype(int)

    d["コンポーネント合格数"] = cnt
    d["合格数区分"] = d["コンポーネント合格数"].map({
        0: "0/3（全弱）",
        1: "1/3（片輪）",
        2: "2/3（概ね良）",
        3: "3/3（万全）",
    })
    return d

def add_race_cv(df_in: pd.DataFrame) -> pd.DataFrame:
    """レース単位のcv (std/mean) を計算して全行に付与"""
    d = df_in.copy()
    key = ["開催日", "場所", "R"]
    if not set(key).issubset(d.columns) or "総合利益度" not in d.columns:
        d["cv"] = np.nan
        return d

    d["総合利益度"] = pd.to_numeric(d["総合利益度"], errors="coerce")

    def _cv(g: pd.DataFrame) -> float:
        vals = g["総合利益度"].dropna().to_numpy()
        if len(vals) == 0:
            return np.nan
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=0)) if len(vals) >= 2 else 0.0
        return float(std / mean) if mean != 0 else np.nan

    cv_map = d.groupby(key, observed=True).apply(_cv).rename("cv").reset_index()
    return d.merge(cv_map, on=key, how="left")