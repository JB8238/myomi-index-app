import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import re
import io
from datetime import datetime


# =========================================================
# 期間分割検証（6:4）
# =========================================================
def split_dates_60_40(df: pd.DataFrame, date_col: str = "開催日") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    df を開催日で昇順に並べ、ユニーク開催日の前半60%をtrain、後半40%をtestに分割する
    """
    if date_col not in df.columns or df.empty:
        return df.copy(), df.iloc[0:0].copy()
    d = df.copy()
    d[date_col] = pd.to_numeric(d[date_col], errors="coerce")
    dates = sorted([x for x in d[date_col].dropna().unique().tolist()])
    if not dates:
        return d.copy(), d.iloc[0:0].copy()
    cut = int(np.ceil(len(dates) * 0.6))
    train_dates = set(dates[:cut])
    test_dates = set(dates[cut:])
    train = d[d[date_col].isin(train_dates)].copy()
    test = d[d[date_col].isin(test_dates)].copy()
    return train, test

def eval_condition_row(df: pd.DataFrame, row: pd.Series, bet_type: str) -> tuple[int, float]:
    """
    条件行(row)に一致するdfのサブセットを作り、件数とROIを返す。
    bet_type: "win" or "place"
    """
    x = df.copy()
    # 必須キー
    for k in ["レースレベル", "上昇値区分", "人気乖離区分"]:
        if k in row.index and k in x.columns:
            x = x[x[k].astype(str) == str(row[k])]
    # 前提フィルタ（存在する場合のみ）
    if "前提_合格数区分" in row.index and "合格数区分" in x.columns:
        prereq = str(row["前提_合格数区分"]).strip()
        if prereq and prereq != "ALL":
            x = x[x["合格数区分"].astype(str) == prereq]
    if "前提_cv閾値" in row.index and "cv" in x.columns:
        th = pd.to_numeric(row["前提_cv閾値"], errors="coerce")
        if pd.notna(th):
            x = x[pd.to_numeric(x["cv"], errors="coerce") >= float(th)]

    n = int(len(x))
    if n == 0:
        return 0, float("nan")

    # ROI: はずれ=0の平均（単勝/複勝）
    if bet_type == "win":
        pay = pd.to_numeric(x.get("単勝"), errors="coerce").fillna(0)
    else:
        pay = pd.to_numeric(x.get("複勝"), errors="coerce").fillna(0)
    roi = float(pay.mean())     # 100円あたり
    return n, roi

def add_stability_flags(
    cond_df: pd.DataFrame,
    df_all: pd.DataFrame,
    bet_type: str,
    min_roi: float,
    min_n: int,
    date_col: str = "開催日",
) -> pd.DataFrame:
    """
    条件表(cond_df)の各行について、開催日6:4分割で検証OK/NGを付与して返す。
    bet_type: "win" or "place"
    """
    if cond_df is None or cond_df.empty:
        return cond_df

    train, test = split_dates_60_40(df_all, date_col=date_col)

    out = cond_df.copy()
    train_n, train_roi, test_n, test_roi, ok = [], [], [], [], []
    for _, r in out.iterrows():
        n1, roi1 = eval_condition_row(train, r, bet_type)
        n2, roi2 = eval_condition_row(test, r, bet_type)
        train_n.append(n1); train_roi.append(roi1)
        test_n.append(n2); test_roi.append(roi2)
        ok.append((n1 >= min_n) and (n2 >= min_n) and (roi1 >= min_roi) and (roi2 >= min_roi))

    out["検証_train件数"] = train_n
    out["検証_trainROI"] = train_roi
    out["検証_test件数"] = test_n
    out["検証_testROI"] = test_roi
    out["期間分割OK"] = ok
    return out


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
# 偏差値合格数 × 混戦度区分（二軸）集計
# =========================================================
def calc_roi_pivot_2d(
    src: pd.DataFrame,
    row_col: str,
    col_col: str,
) -> dict[str, pd.DataFrame]:
    """
    row_col × col_col の2軸で、以下をpivotで返す
        - 頭数
        - 単勝的中率(%)
        - 複勝的中率(%)
        - 単勝回収率(%)
        - 複勝回収率(%)
    戻り値: {"頭数": df, "単勝的中率": df, ...}
    """
    if src is None or src.empty:
        return {}
    if row_col not in src.columns or col_col not in src.columns:
        return {}

    d = src.copy()
    # ROI計算用（はずれ=0）
    d["単勝_回収"] = pd.to_numeric(d.get("単勝"), errors="coerce").fillna(0)
    d["複勝_回収"] = pd.to_numeric(d.get("複勝"), errors="coerce").fillna(0)

    # 的中フラグが無い場合は補完（analysisのdfには基本入っているが安全策）
    if "単勝的中" not in d.columns:
        d["単勝的中"] = pd.to_numeric(d.get("単勝"), errors="coerce").fillna(0) > 0
    if "複勝的中" not in d.columns:
        d["複勝的中"] = pd.to_numeric(d.get("複勝"), errors="coerce").fillna(0) > 0

    # dropna（row/colが欠損だとpivotが崩れるため）
    d = d.dropna(subset=[row_col, col_col])
    if d.empty:
        return {}

    # 集計（2軸groupby）
    g = (
        d.groupby([row_col, col_col], observed=True)
        .agg(
            n=("馬名", "count") if "馬名" in d.columns else ("馬番", "count"),
            win_hit=("単勝的中", "mean"),
            plc_hit=("複勝的中", "mean"),
            win_roi=("単勝_回収", "mean"),
            plc_roi=("複勝_回収", "mean"),
        )
        .reset_index()
    )
    g["win_hit"] = g["win_hit"] * 100
    g["plc_hit"] = g["plc_hit"] * 100

    # pivot化
    out = {
        "頭数": g.pivot_table(index=row_col, columns=col_col, values="n", aggfunc="first", observed=True),
        "単勝的中率": g.pivot_table(index=row_col, columns=col_col, values="win_hit", aggfunc="first", observed=True),
        "複勝的中率": g.pivot_table(index=row_col, columns=col_col, values="plc_hit", aggfunc="first", observed=True),
        "単勝回収率": g.pivot_table(index=row_col, columns=col_col, values="win_roi", aggfunc="first", observed=True),
        "複勝回収率": g.pivot_table(index=row_col, columns=col_col, values="plc_roi", aggfunc="first", observed=True),
    }

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