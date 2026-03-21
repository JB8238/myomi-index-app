import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import re
from datetime import datetime

# =========================================================
# 設定
# =========================================================
DATA_DIR = Path(".", "prof_result")
MERGED_RETURN_PATH = Path("./data/return_data_merged.csv")

RESULT_COLS = [
    "人気",
    "単オッズ",
    "複勝オッズ下",
    "上",
    "単勝",
    "複勝",
    "馬連配当",
]

st.set_page_config(page_title="指数分析", page_icon="📈", layout="wide")
st.title("📈 指数分析ページ")
st.caption("results_prof_index（指数） + return_data（払戻）を統合して分析します（着順不使用）")
st.divider()

# =========================================================
# ユーティリティ
# =========================================================
def extract_yyyymmdd_from_name(filename: str) -> int | None:
    m = re.findall(r"(\d{8})", filename)
    for s in reversed(m):
        try:
            datetime.strptime(s, "%Y%m%d")
            return int(s)
        except ValueError:
            pass
    return None


def list_results_index_files(data_dir: Path):
    return sorted([p for p in data_dir.rglob("results_prof_index_*.csv") if p.is_file()])


@st.cache_data(show_spinner="📥 results_prof_index 読み込み中…")
def load_prof(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp932")
    df.columns = [str(c).strip() for c in df.columns]

    for c in ["R", "馬番", "総合利益度", "総合利益度順位"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "場所" in df.columns:
        df["場所"] = df["場所"].astype(str).str.replace("\u3000", " ").str.strip()
    if "馬名" in df.columns:
        df["馬名"] = df["馬名"].astype(str).str.replace("\u3000", " ").str.strip()

    return df


@st.cache_data(show_spinner="📥 return_data 読み込み中…")
def load_return(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    df.rename(columns={"Ｒ": "R"}, inplace=True)

    keep = [c for c in ["開催日", "場所", "R", "馬番"] + RESULT_COLS if c in df.columns]
    df = df[keep].copy()

    for c in ["開催日", "R", "馬番"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in RESULT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "場所" in df.columns:
        df["場所"] = df["場所"].astype(str).str.replace("\u3000", " ").str.strip()

    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["単勝的中"] = pd.to_numeric(out.get("単勝"), errors="coerce") > 0
    out["複勝的中"] = pd.to_numeric(out.get("複勝"), errors="coerce") > 0

    if {"人気", "総合利益度順位"}.issubset(out.columns):
        out["人気乖離"] = out["人気"] - out["総合利益度順位"]
    else:
        out["人気乖離"] = np.nan

    return out


@st.cache_data(show_spinner="📚 指数履歴を構築中…")
def build_index_history(data_dir: Path) -> pd.DataFrame:
    rows = []
    for p in list_results_index_files(data_dir):
        d = extract_yyyymmdd_from_name(p.name)
        if d is None:
            continue

        df0 = pd.read_csv(p, encoding="cp932")
        df0.columns = [str(c).strip() for c in df0.columns]
        if not {"馬名", "総合利益度"}.issubset(df0.columns):
            continue

        tmp = df0[["馬名", "総合利益度"]].copy()
        tmp["馬名"] = tmp["馬名"].astype(str).str.replace("\u3000", " ").str.strip()
        tmp["総合利益度"] = pd.to_numeric(tmp["総合利益度"], errors="coerce")
        tmp["開催日"] = d
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["馬名", "総合利益度", "開催日"])

    return pd.concat(rows, ignore_index=True)


def attach_prev_total(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    cur = df.copy()
    cur["開催日"] = pd.to_numeric(cur["開催日"], errors="coerce")

    hist2 = hist.rename(columns={"総合利益度": "前走総合利益度"}).copy()
    hist2["開催日"] = pd.to_numeric(hist2["開催日"], errors="coerce")

    # ★ merge_asof のために onキー最優先でソート
    cur = cur.sort_values(["開催日", "馬名"])
    hist2 = hist2.sort_values(["開催日", "馬名"])

    return pd.merge_asof(
        cur,
        hist2,
        by="馬名",
        on="開催日",
        direction="backward",
        allow_exact_matches=False,
    )


# =========================================================
# データ読み込み
# =========================================================
files = list_results_index_files(DATA_DIR)
if not files:
    st.error("results_prof_index CSV が見つかりません")
    st.stop()

dated = [(extract_yyyymmdd_from_name(p.name), p) for p in files]
dated = [(d, p) for d, p in dated if d is not None]
dated.sort()

available_dates = [d for d, _ in dated]

with st.sidebar:
    mode = st.radio("対象期間", ["最新日だけ", "日付を選ぶ", "全期間"], index=0)
    if mode == "最新日だけ":
        target_dates = [available_dates[-1]]
    elif mode == "日付を選ぶ":
        target_dates = st.multiselect("開催日", available_dates, default=[available_dates[-1]])
    else:
        target_dates = available_dates

# =========================================================
# 統合
# =========================================================
frames = []
df_ret = load_return(MERGED_RETURN_PATH) if MERGED_RETURN_PATH.exists() else None

for d, p in dated:
    if d not in target_dates:
        continue

    dfp = load_prof(p)
    dfp["開催日"] = d

    if df_ret is not None:
        dfp = dfp.merge(
            df_ret[df_ret["開催日"] == d],
            on=["開催日", "場所", "R", "馬番"],
            how="left",
        )

    frames.append(dfp)

df = pd.concat(frames, ignore_index=True)

# =========================================================
# 前走付与 → 上昇値計算
# =========================================================
history = build_index_history(DATA_DIR)
df = attach_prev_total(df, history)

df["利益度上昇値"] = (
    pd.to_numeric(df["総合利益度"], errors="coerce")
    - pd.to_numeric(df["前走総合利益度"], errors="coerce")
)

df = add_derived_columns(df)


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


st.header("① 利益度上昇値 × 回収率・的中率")
if df["利益度上昇値"].notna().sum() == 0:
    st.info("利益度上昇値が算出できるデータがありません。")
else:
    bins_up = [-999, 0, 5, 10, 999]
    labels_up = ["<=0", "1–5", "6–10", "11+"]
    df["上昇値区分"] = pd.cut(df["利益度上昇値"], bins=bins_up, labels=labels_up, include_lowest=True)
    t1 = calc_roi_table(df.dropna(subset=["上昇値区分"]), "上昇値区分")
    st.dataframe(
        t1.style.format({
            "単勝的中率": "{:.1f}%",
            "複勝的中率": "{:.1f}%",
            "単勝回収率": "{:.1f}%",
            "複勝回収率": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )


st.header("② 人気乖離（人気−総合利益度順位）× 回収率・的中率")
if df["人気乖離"].notna().sum() == 0:
    st.info("人気乖離が算出できるデータがありません（人気 or 総合利益度順位が不足）。")
else:
    bins_gap = [-999, -5, -1, 3, 7, 999]
    labels_gap = ["<=-5", "-4〜-1", "0〜3", "4〜7", "8+"]
    df["人気乖離区分"] = pd.cut(df["人気乖離"], bins=bins_gap, labels=labels_gap, include_lowest=True)
    t2 = calc_roi_table(df.dropna(subset=["人気乖離区分"]), "人気乖離区分")
    st.dataframe(
        t2.style.format({
            "単勝的中率": "{:.1f}%",
            "複勝的中率": "{:.1f}%",
            "単勝回収率": "{:.1f}%",
            "複勝回収率": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# デバッグ表示
# =========================================================
st.write(
    "前走総合利益度 notna:",
    df["前走総合利益度"].notna().sum(),
    "/",
    len(df),
)
st.write(
    "利益度上昇値 notna:",
    df["利益度上昇値"].notna().sum(),
    "/",
    len(df),
)

st.dataframe(df.head(50), use_container_width=True)