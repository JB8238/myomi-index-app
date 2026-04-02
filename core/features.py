import pandas as pd
import numpy as np


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

def add_race_cv_local(d: pd.DataFrame) -> pd.DataFrame:
    """単一レース用：総合利益度からcv(std/mean)を計算して全行に付与"""
    x = d.copy()
    if "総合利益度" not in x.columns:
        x["cv"] = np.nan
        return x
    vals = pd.to_numeric(x["総合利益度"], errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        x["cv"] = np.nan
        return x
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=0)) if len(vals) >= 2 else 0.0
    x["cv"] = float(std / mean) if mean != 0 else np.nan
    return x

def add_race_deviation_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    key = ["開催日", "場所", "R"]

    targets = {
        "総合利益度": "総合偏差値",
        "騎手利益度": "騎手偏差値",
        "種牡馬利益度": "種牡馬偏差値",
        "調教師利益度": "調教師偏差値",
    }

    for src, dst in targets.items():
        d[src] = pd.to_numeric(d.get(src), errors="coerce")

        def _dev(g):
            x = g[src]
            mu = x.mean()
            sd = x.std(ddof=0)
            if pd.isna(sd) or sd == 0:
                return pd.Series(50.0, index=g.index)
            return 50 + 10 * (x - mu) / sd

        d[dst] = d.groupby(key, observed=True).apply(_dev).reset_index(level=key, drop=True)

    return d

def add_deviation_component_pass(df: pd.DataFrame, threshold: float = 60.0) -> pd.DataFrame:
    d = df.copy()

    cols = ["騎手偏差値", "種牡馬偏差値", "調教師偏差値"]
    cnt = 0
    for c in cols:
        if c in d.columns:
            cnt += (pd.to_numeric(d[c], errors="coerce") >= threshold).astype(int)

    d["偏差値合格数"] = cnt
    d["偏差値合格数区分"] = d["偏差値合格数"].map({
        0: "0/3",
        1: "1/3",
        2: "2/3",
        3: "3/3",
    })
    return d