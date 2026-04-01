import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import re
import io
from datetime import datetime

from core.features import add_component_pass_count, add_race_cv
from core.history import build_index_history, attach_prev_total
from core.loaders import load_preprocessed, load_return
from core.aggregations import calc_roi_table, make_heatmap_table, build_condition_cells, extract_buy_conditions_win, extract_buy_conditions_place

# =========================================================
# 設定
# =========================================================
DATA_DIR = Path("prof_result")
PREP_DIR = Path("data")
MERGED_RETURN_PATH = Path("./data/return_data_merged.csv")
BUY_WIN_FULL_PATH = Path("./data/buy_conditions_full_win.csv")
BUY_PLC_FULL_PATH = Path("./data/buy_conditions_full_place.csv")


# RESULT_COLS = [
#     "人気",
#     "単オッズ",
#     "複勝オッズ下",
#     "上",
#     "単勝",
#     "複勝",
#     "馬連配当",
# ]

st.set_page_config(page_title="指数分析", page_icon="📈", layout="wide")
st.title("📈 指数分析ページ")
st.caption("results_prof_index（指数） + return_data（払戻）を統合して分析します（着順不使用）")
st.divider()

# --------------------------------------
# Homeに戻るリンク（ページ上部に表示）
# --------------------------------------
st.page_link(
    "app.py",
    label="← 開催レース一覧へ戻る",
    icon="🏠",
    use_container_width=True,
)
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

# def list_preprocessed_files(data_dir: Path):
#     return sorted([p for p in data_dir.rglob("preprocessed_data_*.csv") if p.is_file()])

# @st.cache_data(show_spinner="📥 preprocessed_data 読み込み中…")
# def load_preprocessed(data_dir: Path) -> pd.DataFrame:
#     rows = []
#     for p in list_preprocessed_files(data_dir):
#         d = extract_yyyymmdd_from_name(p.name)
#         if d is None:
#             continue

#         df0 = pd.read_csv(p, encoding="utf-8")
#         df0.columns = [str(c).strip() for c in df0.columns]

#         if not {"場所", "R", "馬番", "レースレベル"}.issubset(df0.columns):
#             continue

#         tmp = df0[["場所", "R", "馬番", "レースレベル"]].copy()
#         tmp["開催日"] = d

#         # 正規化
#         tmp["場所"] = tmp["場所"].astype(str).str.replace("\u3000", " ").str.strip()
#         tmp["レースレベル"] = tmp["レースレベル"].astype(str).str.strip()

#         tmp["R"] = pd.to_numeric(tmp["R"], errors="coerce")
#         tmp["馬番"] = pd.to_numeric(tmp["馬番"], errors="coerce")

#         rows.append(tmp)

#     if not rows:
#         return pd.DataFrame(columns=["開催日", "場所", "R", "馬番", "レースレベル"])

#     return pd.concat(rows, ignore_index=True)

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


# @st.cache_data(show_spinner="📥 return_data 読み込み中…")
# def load_return(path: Path) -> pd.DataFrame:
#     df = pd.read_csv(path, encoding="utf-8-sig")
#     df.columns = [str(c).strip() for c in df.columns]

#     df.rename(columns={"Ｒ": "R"}, inplace=True)

#     keep = [c for c in ["開催日", "場所", "R", "馬番"] + RESULT_COLS if c in df.columns]
#     df = df[keep].copy()

#     for c in ["開催日", "R", "馬番"]:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")

#     for c in RESULT_COLS:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")

#     if "場所" in df.columns:
#         df["場所"] = df["場所"].astype(str).str.replace("\u3000", " ").str.strip()

#     return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["単勝的中"] = pd.to_numeric(out.get("単勝"), errors="coerce") > 0
    out["複勝的中"] = pd.to_numeric(out.get("複勝"), errors="coerce") > 0

    if {"人気", "総合利益度順位"}.issubset(out.columns):
        out["人気乖離"] = out["人気"] - out["総合利益度順位"]
    else:
        out["人気乖離"] = np.nan

    return out


# @st.cache_data(show_spinner="📚 指数履歴を構築中…")
# def build_index_history(data_dir: Path) -> pd.DataFrame:
#     rows = []
#     for p in list_results_index_files(data_dir):
#         d = extract_yyyymmdd_from_name(p.name)
#         if d is None:
#             continue

#         df0 = pd.read_csv(p, encoding="cp932")
#         df0.columns = [str(c).strip() for c in df0.columns]
#         if not {"馬名", "総合利益度"}.issubset(df0.columns):
#             continue

#         tmp = df0[["馬名", "総合利益度"]].copy()
#         tmp["馬名"] = tmp["馬名"].astype(str).str.replace("\u3000", " ").str.strip()
#         tmp["総合利益度"] = pd.to_numeric(tmp["総合利益度"], errors="coerce")
#         tmp["開催日"] = d
#         rows.append(tmp)

#     if not rows:
#         return pd.DataFrame(columns=["馬名", "総合利益度", "開催日"])

#     return pd.concat(rows, ignore_index=True)


# def attach_prev_total(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
#     cur = df.copy()
#     cur["開催日"] = pd.to_numeric(cur["開催日"], errors="coerce")

#     hist2 = hist.rename(columns={"総合利益度": "前走総合利益度"}).copy()
#     hist2["開催日"] = pd.to_numeric(hist2["開催日"], errors="coerce")

#     # ★ merge_asof のために onキー最優先でソート
#     cur = cur.sort_values(["開催日", "馬名"])
#     hist2 = hist2.sort_values(["開催日", "馬名"])

#     return pd.merge_asof(
#         cur,
#         hist2,
#         by="馬名",
#         on="開催日",
#         direction="backward",
#         allow_exact_matches=False,
#     )

def make_bin_bounds(bins: list[float], labels: list[str], key_name: str) -> pd.DataFrame:
    """
    pd.cut の bins/labels から (label -> low/high/include_lowest) の対応表を作る
    ルール: 最初のbinのみ low を含む（include_lowest=True相当）
    """
    rows = []
    for i, lab in enumerate(labels):
        rows.append({
            key_name: lab,
            "low": float(bins[i]),
            "high": float(bins[i+1]),
            "include_lowest": True if i == 0 else False,
        })
    return pd.DataFrame(rows)

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")

def calc_race_competitiveness(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    レース単位の混戦度（指数分散）を計算
    key: 開催日, 場所, R
    指標:
        - cv = std/mean（平均が小さいと大きくなりやすいので分位推奨）
        - gap12 = top1 - top2（上位の抜け具合）
        - std = 標準偏差（混戦度そのもの）
    """
    key = ["開催日", "場所", "R"]
    need = set(key + ["総合利益度"])
    if not need.issubset(df_in.columns):
        return pd.DataFrame(columns=key + ["n","mean","std","cv","top1", "top2", "gap12"])

    x = df_in.copy()
    x["総合利益度"] = pd.to_numeric(x["総合利益度"], errors="coerce")
    x = x.dropna(subset=["総合利益度"])

    def _agg(g: pd.DataFrame) -> pd.Series:
        vals = g["総合利益度"].sort_values(ascending=False).to_numpy()
        n = len(vals)
        top1 = float(vals[0]) if n >= 1 else np.nan
        top2 = float(vals[1]) if n >= 2 else np.nan
        gap12 = (top1 - top2) if pd.notna(top1) and pd.notna(top2) else np.nan
        mean = float(np.nanmean(vals)) if n else np.nan
        std = float(np.nanstd(vals, ddof=0)) if n >= 2 else 0.0
        cv = float(std / mean) if pd.notna(mean) and mean != 0 else np.nan
        return pd.Series({
            "n": n,
            "mean": mean,
            "std": std,
            "cv": cv,
            "top1": top1,
            "top2": top2,
            "gap12": gap12,
        })

    out = x.groupby(key, observed=True).apply(_agg).reset_index()
    return out

# def add_component_pass_count(df_in: pd.DataFrame) -> pd.DataFrame:
#     """
#     総合利益度>=0の馬について、
#     騎手/調教師/種牡馬 利益度が >=0 のカテゴリ数を数える
#     """
#     d = df_in.copy()

#     def _ok(x):
#         return pd.notna(x) and x >= 0

#     cnt = 0
#     for col in ["騎手利益度", "調教師利益度", "種牡馬利益度"]:
#         if col in d.columns:
#             d[col] = pd.to_numeric(d[col], errors="coerce")
#             cnt += d[col].apply(_ok).astype(int)

#     d["コンポーネント合格数"] = cnt
#     d["合格数区分"] = d["コンポーネント合格数"].map({
#         0: "0/3（全弱）",
#         1: "1/3（片輪）",
#         2: "2/3（概ね良）",
#         3: "3/3（万全）",
#     })
#     return d

# def add_race_cv(df_in: pd.DataFrame) -> pd.DataFrame:
#     """レース単位のcv (std/mean) を計算して全行に付与"""
#     d = df_in.copy()
#     key = ["開催日", "場所", "R"]
#     if not set(key).issubset(d.columns) or "総合利益度" not in d.columns:
#         d["cv"] = np.nan
#         return d

#     d["総合利益度"] = pd.to_numeric(d["総合利益度"], errors="coerce")

#     def _cv(g: pd.DataFrame) -> float:
#         vals = g["総合利益度"].dropna().to_numpy()
#         if len(vals) == 0:
#             return np.nan
#         mean = float(np.mean(vals))
#         std = float(np.std(vals, ddof=0)) if len(vals) >= 2 else 0.0
#         return float(std / mean) if mean != 0 else np.nan

#     cv_map = d.groupby(key, observed=True).apply(_cv).rename("cv").reset_index()
#     return d.merge(cv_map, on=key, how="left")


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
    mode = st.radio(
        "対象期間",
        ["最新日だけ", "日付を選ぶ", "期間指定", "全期間"],
        index=0
    )
    if mode == "最新日だけ":
        target_dates = [available_dates[-1]]
    elif mode == "日付を選ぶ":
        target_dates = st.multiselect(
            "開催日", available_dates,
            default=[available_dates[-1]]
        )
    elif mode == "期間指定":
        date_objs = [
            datetime.strptime(str(d), "%Y%m%d").date()
            for d in available_dates
        ]
        start_date, end_date = st.date_input(
            "対象期間（開始～終了）",
            value=(min(date_objs), max(date_objs)),
            min_value=min(date_objs),
            max_value=max(date_objs),
        )
        target_dates = [
            d for d in available_dates
            if start_date
            <= datetime.strptime(str(d), "%Y%m%d").date()
            <= end_date
        ]
        if not target_dates:
            st.warning("指定期間に該当する開催日がありません。")
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
# レースレベル（Lv1～Lv5）を付与
# =========================================================
df_pre = load_preprocessed(PREP_DIR)

df = df.merge(
    df_pre,
    on=["開催日", "場所", "R", "馬番"],
    how="left",
)

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

with st.sidebar:
    st.subheader("レースレベル（Lv）")
    
    if "レースレベル" in df.columns:
        all_lv = sorted(
            df["レースレベル"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_lv = st.multiselect(
            "対象レースレベル",
            options=all_lv,
            default=all_lv,   # デフォルトは全選択
        )
    else:
        selected_lv = []
        st.caption("レースレベル情報がありません")

    # 分析の母集団（合格馬フィルタ）切り替え
    st.subheader("分析母集団")
    population_mode = st.radio(
        "母集団を選択",
        ["全馬（フィルタなし）", "合格馬のみ（総合利益度>=0）"],
        index=0,
        horizontal=False,
    )

    st.subheader("戦略前提フィルタ（④×⑤）")
    # 3/3固定（チェックでON/OFFにすると運用が楽）
    use_comp33 = st.checkbox("合格数区分を 3/3（万全）に固定", value=True)
    # 混戦度上位率（デフォルト50%）
    top_rate = st.slider("混戦度（cv）上位率", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    # 買い条件 抽出パラメータ（UI）
    st.subheader("買い条件 抽出パラメータ")

    st.markdown("**単勝条件**")
    win_min_roi = st.slider(
        "単勝ROI 下限 (%)",
        min_value=90.0,
        max_value=150.0,
        value=110.0,
        step=1.0,
    )
    win_min_n = st.slider(
        "単勝 件数 下限",
        min_value=3,
        max_value=50,
        value=10,
        step=1,
    )

    st.markdown("**複勝条件**")
    plc_min_roi = st.slider(
        "複勝ROI 下限 (%)",
        min_value=90.0,
        max_value=130.0,
        value=105.0,
        step=1.0,
    )
    plc_min_n = st.slider(
        "複勝 件数 下限",
        min_value=3,
        max_value=50,
        value=10,
        step=1,
    )


# # =========================================================
# # 分析表示（回収率/的中率）
# # =========================================================
# def calc_roi_table(src: pd.DataFrame, group_col: str) -> pd.DataFrame:
#     """
#     回収率は「外れ=0円」で平均（単勝/複勝は100円あたりの払戻円を想定）
#     的中率は bool の平均（%表示用に×100）
#     """
#     d = src.copy()
#     # 外れ（NaN）を 0 として回収率計算
#     d["単勝_回収"] = pd.to_numeric(d.get("単勝"), errors="coerce").fillna(0)
#     d["複勝_回収"] = pd.to_numeric(d.get("複勝"), errors="coerce").fillna(0)

#     out = (
#         d.groupby(group_col, observed=True)
#         .agg(
#             頭数=("馬名", "count") if "馬名" in d.columns else ("馬番", "count"),
#             単勝的中率=("単勝的中", "mean"),
#             複勝的中率=("複勝的中", "mean"),
#             単勝回収率=("単勝_回収", "mean"),
#             複勝回収率=("複勝_回収", "mean"),
#         )
#         .reset_index()
#     )

#     # 表示用（%）
#     out["単勝的中率"] = out["単勝的中率"] * 100
#     out["複勝的中率"] = out["複勝的中率"] * 100
#     # 単勝回収率/複勝回収率は「円/100円」なので数値=％相当（例: 95.2 → 95.2%）

#     return out

# ---- レースレベルによるフィルタリング ----
df_filtered = df.copy()
if selected_lv and "レースレベル" in df_filtered.columns:
    df_filtered = df_filtered[
        df_filtered["レースレベル"].astype(str).isin(selected_lv)
    ]

# 母集団フィルタ（合格馬のみ）
if population_mode == "合格馬のみ（総合利益度>=0）" and "総合利益度" in df_filtered.columns:
    df_filtered["総合利益度"] = pd.to_numeric(df_filtered["総合利益度"], errors="coerce")
    df_filtered = df_filtered[df_filtered["総合利益度"].notna() & (df_filtered["総合利益度"] >= 0)]

st.caption(
    f"レースレベル絞り込み後: {len(df_filtered)} / {len(df)}"
)
st.caption(f"分析母集団: {population_mode}")

# ---- レース単位の指数集中度/分散を付与 ----
race_stats = calc_race_competitiveness(df_filtered)
df_filtered = add_component_pass_count(df_filtered)
# df_filtered = df_filtered.merge(race_stats, on=["開催日", "場所", "R"], how="left")
df_filtered = add_race_cv(df_filtered)

with st.sidebar:
    st.subheader("混戦度フィルタ（指数の分散）")
    # cv: 変動係数（std/mean）を推奨。ない場合に備えてガード
    if "cv" in df_filtered.columns and df_filtered["cv"].dropna().size > 0:
        cv_min = float(np.nanmin(df_filtered["cv"]))
        cv_max = float(np.nanmax(df_filtered["cv"]))
        # 極端値のときの保険
        if not np.isfinite(cv_min): cv_min = 0.0
        if not np.isfinite(cv_max): cv_max = 1.0
        cv_range = st.slider(
            "対象 cv (std/mean) 範囲",
            min_value=float(max(0.0, cv_min)),
            max_value=float(max(0.01, cv_max)),
            value=(float(max(0.0, cv_min)), float(max(0.01, cv_max))),
            step=0.01,
        )
    else:
        cv_range = None
        st.caption("cv（混戦度）を計算できるデータがありません")
        
    st.subheader("混戦度（レース単位）")
    comp_metric = st.radio(
        "混戦度の指標",
        ["cv (std/mean)", "std（標準偏差）", "gap12（1位-2位差）"],
        index=0
    )
    q_bins = st.slider("混戦度の分位数 (qcut)", min_value=4, max_value=12, value=8, step=1)


# ---- 分析実行 ----
df_hm = df_filtered.copy()
metric_col = {"cv (std/mean)": "cv", "std（標準偏差）": "std", "gap12（1位-2位差）": "gap12"}[comp_metric]

if metric_col in df_hm.columns and df_hm[metric_col].dropna().size > 0:
    # qcutは同値が多いと落ちることがあるので duplicates='drop' で安全に
    df_hm["混戦度区分"] = pd.qcut(df_hm[metric_col], q=q_bins, duplicates="drop")
else:
    df_hm["混戦度区分"] = np.nan

if "総合利益度" in df_hm.columns and df_hm["総合利益度"].dropna().size > 0:
    df_hm["指数値区分"] = pd.qcut(pd.to_numeric(df_hm["総合利益度"], errors="coerce"), q=6, duplicates="drop")
else:
    df_hm["指数値区分"] = np.nan

if "総合利益度順位" in df_hm.columns:
    r = pd.to_numeric(df_hm["総合利益度順位"], errors="coerce")
    bins_rank = [-999, 3, 6, 10, 999]
    labels_rank = ["1-3", "4-6", "7-10", "11+"]
    df_hm["指数順位区分"] = pd.cut(r, bins=bins_rank, labels=labels_rank, include_lowest=True)
else:
    df_hm["指数順位区分"] = np.nan

if "総合利益度" in df_hm.columns:
    s = pd.to_numeric(df_hm["総合利益度"], errors="coerce")
    bins_index = [-999, -10, -5, 0, 5, 10, 15, 999]
    labels_index = ["<=-10", "(-10,-5]", "(-5,0]", "(0,5]", "(5,10]", "(10,15]", "15+"]
    df_hm["指数値区分"] = pd.cut(r, bins=bins_index, labels=labels_index, include_lowest=True)
else:
    df_hm["指数値区分"] = np.nan

# df_hm["コンポーネント合格数"] = df_hm["コンポーネント合格数"].astype("Int64")
# df_hm["合格数区分"] = df_hm["コンポーネント合格数"].map({
#     0: "0/3（全弱）",
#     1: "1/3（片輪）",
#     2: "2/3（概ね良）",
#     3: "3/3（万全）",
# })

if use_comp33 and "合格数区分" in df_hm.columns:
    df_hm = df_hm[df_hm["合格数区分"] == "3/3（万全）"]

# cv上位率（レース単位で閾値を出して固定）
cv_threshold = None
if "cv" in df_hm.columns:
    race_key = ["開催日", "場所", "R"]
    race_cv = df_hm.drop_duplicates(race_key)[race_key + ["cv"]].dropna()
    if not race_cv.empty:
        cv_threshold = float(race_cv["cv"].quantile(1 - top_rate))
        df_hm = df_hm[df_hm["cv"] >= cv_threshold]

st.caption(f"戦略前提フィルタ後: {len(df_hm)} / {len(df_filtered)}")
if cv_threshold is not None:
    st.caption(f"混戦度(cv) 閾値: {cv_threshold:.3f} (上位{int(top_rate*100)}%)")
    

# ----- 分析内容 -----
st.header("① 利益度上昇値 × 回収率・的中率")
if df_hm["利益度上昇値"].notna().sum() == 0:
    st.info("利益度上昇値が算出できるデータがありません。")
else:
    bins_up = [-999, 0, 2, 4, 6, 8, 10, 14, 999]
    labels_up = ["<=0", "(0,2]", "(2,4]", "(4,6]", "(6,8]", "(8,10]", "(10,14]", "14+"]
    df_hm["上昇値区分"] = pd.cut(
        df_hm["利益度上昇値"], bins=bins_up, labels=labels_up, include_lowest=True
    )
    t1 = calc_roi_table(df_hm.dropna(subset=["上昇値区分"]), "上昇値区分")
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
if df_hm["人気乖離"].notna().sum() == 0:
    st.info("人気乖離が算出できるデータがありません（人気 or 総合利益度順位が不足）。")
else:
    bins_gap = [-999, -5, -1, 3, 7, 999]
    labels_gap = ["<=-5", "-4〜-1", "0〜3", "4〜7", "8+"]
    df_hm["人気乖離区分"] = pd.cut(
        df_hm["人気乖離"], bins=bins_gap, labels=labels_gap, include_lowest=True
    )
    t2 = calc_roi_table(df_hm.dropna(subset=["人気乖離区分"]), "人気乖離区分")
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

st.header("③-1 指数順位 × 混戦度区分 (qcut) × 回収率")

x_rank = df_hm.dropna(subset=["混戦度区分", "指数順位区分"]).copy()
x_rank["単勝_回収"] = pd.to_numeric(x_rank.get("単勝"), errors="coerce").fillna(0)
x_rank["複勝_回収"] = pd.to_numeric(x_rank.get("複勝"), errors="coerce").fillna(0)

pivot_win_rank = x_rank.pivot_table(index="指数順位区分", columns="混戦度区分", values="単勝_回収", aggfunc="mean", observed=True)
pivot_plc_rank = x_rank.pivot_table(index="指数順位区分", columns="混戦度区分", values="複勝_回収", aggfunc="mean", observed=True)
pivot_n_rank = x_rank.pivot_table(index="指数順位区分", columns="混戦度区分", values="馬番" if "馬番" in x_rank.columns else "馬名", aggfunc="count", observed=True, fill_value=0)

tab = st.radio("表示", ["単勝回収率", "複勝回収率", "件数"], horizontal=True)
if tab == "単勝回収率":
    st.dataframe(pivot_win_rank.style.format("{:.1f}%"), use_container_width=True)
elif tab == "複勝回収率":
    st.dataframe(pivot_plc_rank.style.format("{:.1f}%"), use_container_width=True)
else:
    st.dataframe(pivot_n_rank.style.format("{:.0f}"), use_container_width=True)


st.header("③-2 指数値 × 混戦度区分 (qcut) × 回収率")

x_index = df_hm.dropna(subset=["混戦度区分", "指数値区分"]).copy()
x_index["単勝_回収"] = pd.to_numeric(x_index.get("単勝"), errors="coerce").fillna(0)
x_index["複勝_回収"] = pd.to_numeric(x_index.get("複勝"), errors="coerce").fillna(0)

pivot_win_index = x_index.pivot_table(index="指数値区分", columns="混戦度区分", values="単勝_回収", aggfunc="mean", observed=True)
pivot_plc_index = x_index.pivot_table(index="指数値区分", columns="混戦度区分", values="複勝_回収", aggfunc="mean", observed=True)
pivot_n_index = x_index.pivot_table(index="指数値区分", columns="混戦度区分", values="馬番" if "馬番" in x_index.columns else "馬名", aggfunc="count", observed=True, fill_value=0)

# tab_index = st.radio("表示", ["単勝回収率", "複勝回収率", "件数"], horizontal=True)
if tab == "単勝回収率":
    st.dataframe(pivot_win_index.style.format("{:.1f}%"), use_container_width=True)
elif tab == "複勝回収率":
    st.dataframe(pivot_plc_index.style.format("{:.1f}%"), use_container_width=True)
else:
    st.dataframe(pivot_n_index.style.format("{:.0f}"), use_container_width=True)

# この後のヒートマップ描画は区分が両方揃っている行のみを対象
df_hm = df_hm.dropna(subset=["上昇値区分", "人気乖離区分"])

# # =========================================================
# # ヒートマップによる集計
# # =========================================================
# def make_heatmap_table(d: pd.DataFrame, value_col: str) -> pd.DataFrame:
#     """
#     value_col:
#         - '単勝ROI' / '複勝ROI' / '単勝的中率' / '複勝的中率' / '件数'
#     """
#     x = d.copy()

#     # 区分列が無い時は落ちないように
#     if "上昇値区分" not in x.columns or "人気乖離区分" not in x.columns:
#         return pd.DataFrame()

#     # ROI用に「外れ=0」を作る（NaNの平均を避ける）
#     x["単勝_回収"] = pd.to_numeric(x.get("単勝"), errors="coerce").fillna(0)
#     x["複勝_回収"] = pd.to_numeric(x.get("複勝"), errors="coerce").fillna(0)

#     pivot_count = x.pivot_table(
#         index="上昇値区分", columns="人気乖離区分",
#         values="馬番" if "馬番" in x.columns else "馬名",
#         aggfunc="count", fill_value=0, observed=True
#     )

#     pivot_win_roi = x.pivot_table(
#         index="上昇値区分", columns="人気乖離区分",
#         values="単勝_回収", aggfunc="mean", observed=True
#     )

#     pivot_plc_roi = x.pivot_table(
#         index="上昇値区分", columns="人気乖離区分",
#         values="複勝_回収", aggfunc="mean", observed=True
#     )

#     pivot_win_hit = x.pivot_table(
#         index="上昇値区分", columns="人気乖離区分",
#         values="単勝的中", aggfunc="mean", observed=True
#     ) * 100

#     pivot_plc_hit = x.pivot_table(
#         index="上昇値区分", columns="人気乖離区分",
#         values="複勝的中", aggfunc="mean", observed=True
#     ) * 100

#     if value_col == "件数":
#         return pivot_count
#     if value_col == "単勝ROI":
#         return pivot_win_roi
#     if value_col == "複勝ROI":
#         return pivot_plc_roi
#     if value_col == "単勝的中率":
#         return pivot_win_hit
#     if value_col == "複勝的中率":
#         return pivot_plc_hit

#     return pd.DataFrame()

st.header("④ コンポーネント合格数 × 回収率・的中率")

if "コンポーネント合格数" not in df_filtered.columns:
    st.info("コンポーネント利益度の列が不足しています。")
else:
    df_view = df_filtered.copy()

    # 表示用ラベル
    df_view["合格数区分"] = df_view["コンポーネント合格数"].map({
        0: "0/3（全弱）",
        1: "1/3（片輪）",
        2: "2/3（概ね良）",
        3: "3/3（万全）",
    })

    t_comp = calc_roi_table(
        df_view.dropna(subset=["合格数区分"]),
        "合格数区分"
    )

    st.dataframe(
        t_comp.style.format({
            "単勝的中率": "{:.1f}%",
            "複勝的中率": "{:.1f}%",
            "単勝回収率": "{:.1f}%",
            "複勝回収率": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption("※ 総合利益度>=0 の馬のみを対象")

st.header("⑤ 混戦度 × コンポーネント合格数")

x_ns = df_hm.dropna(subset=["混戦度区分", "合格数区分"]).copy()

x_ns["単勝_回収"] = pd.to_numeric(x_ns.get("単勝"), errors="coerce").fillna(0)
x_ns["複勝_回収"] = pd.to_numeric(x_ns.get("複勝"), errors="coerce").fillna(0)

pivot_win_ns = x_ns.pivot_table(
    index="合格数区分", columns="混戦度区分", values="単勝_回収",
    aggfunc="mean", observed=True,
)

pivot_plc_ns = x_ns.pivot_table(
    index="合格数区分", columns="混戦度区分", values="複勝_回収",
    aggfunc="mean", observed=True,
)

pivot_n_ns = x_ns.pivot_table(
    index="合格数区分", columns="混戦度区分",
    values="馬番" if "馬番" in x_ns.columns else "馬名",
    aggfunc="count", observed=True, fill_value=0,
)

tab = st.radio("表示指標", ["単勝ROI", "複勝ROI", "件数"], horizontal=True)

if tab == "単勝ROI":
    st.dataframe(pivot_win_ns.style.format("{:.1f}%"), use_container_width=True)
elif tab == "複勝ROI":
    st.dataframe(pivot_plc_ns.style.format("{:.1f}%"), use_container_width=True)
else:
    st.dataframe(pivot_n_ns.style.format("{:.0f}"), use_container_width=True)

st.header("⑥ ヒートマップ：Lv × 上昇値区分 × 人気乖離区分")

# 表示する指標を選択（ROI / 的中率 / 件数）
metric = st.radio(
    "表示する指標",
    ["単勝ROI", "複勝ROI", "単勝的中率", "複勝的中率", "件数"],
    horizontal=True
)

hm = make_heatmap_table(df_hm, metric)

if hm.empty:
    st.info("ヒートマップ表示に必要なデータが不足しています（区分列が作れない／データが空）。")
else:
    # 件数が少ないセルは薄くしたいので、件数も同時に取得
    hm_n = make_heatmap_table(df_hm, "件数")

    def colorize(val, n):
        # nが少ない場合は薄く
        if pd.isna(val):
            return "background-color: #f5f5f5; color: #999;"
        # ROI/的中率に応じて色（シンプル）
        # ROI: 100が基準、的中率は高いほど
        if metric in ("単勝ROI", "複勝ROI"):
            base = 100.0
            diff = float(val) - base
            # 強調色（青→赤）
            if diff >= 20:
                color = "rgba(255, 99, 71, 0.35)"   # 強い
            elif diff >= 0:
                color = "rgba(255, 165, 0, 0.25)"   # ちょいプラス
            elif diff <= -20:
                color = "rgba(135, 206, 250, 0.35)"  # 強いマイナス
            else:
                color = "rgba(135, 206, 250, 0.20)"  # ちょいマイナス
        else:
            # 的中率/件数
            if metric == "件数":
                color = "rgba(160, 160, 160, 0.12)"
            else:
                # 的中率が高いほど赤寄り
                if float(val) >= 30:
                    color = "rgba(255, 99, 71, 0.35)"
                elif float(val) >= 15:
                    color = "rgba(255, 165, 0, 0.25)"
                else:
                    color = "rgba(135, 206, 250, 0.20)"

        # 件数で透明度調整
        nn = int(n) if pd.notna(n) else 0
        if nn < 5:
            # サンプル少ないなら薄く
            return f"background-color: {color}; opacity: 0.55;"
        return f"background-color: {color};"

    # Stylerに件数も渡してセルごとに色付け
    def apply_styles(data):
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for r in data.index:
            for c in data.columns:
                styles.loc[r, c] = colorize(data.loc[r, c], hm_n.loc[r, c] if (r in hm_n.index and c in hm_n.columns) else 0)
        return styles

    if metric == "件数":
        sty = hm.style.apply(apply_styles, axis=None).format("{:.0f}")
    else:
        sty = hm.style.apply(apply_styles, axis=None).format("{:.1f}")

    st.caption("※ サンプル数が少ないセル（件数<5）は薄く表示します。")
    st.dataframe(sty, use_container_width=True, height=480)

# # =========================================================
# # 買い条件自動抽出
# # =========================================================
# def build_condition_cells(df_hm: pd.DataFrame) -> pd.DataFrame:
#     x = df_hm.copy()

#     # ROI計算用（はずれ=0）
#     x["単勝_回収"] = pd.to_numeric(x.get("単勝"), errors="coerce").fillna(0)
#     x["複勝_回収"] = pd.to_numeric(x.get("複勝"), errors="coerce").fillna(0)

#     agg = (
#         x.groupby(
#             ["レースレベル", "上昇値区分", "人気乖離区分"],
#             observed=True
#         )
#         .agg(
#             件数=("馬番", "count"),
#             単勝ROI=("単勝_回収", "mean"),
#             単勝的中率=("単勝的中", "mean"),
#             複勝ROI=("複勝_回収", "mean"),
#             複勝的中率=("複勝的中", "mean"),
#         )
#         .reset_index()
#     )

#     agg["単勝的中率"] = agg["単勝的中率"] * 100
#     agg["複勝的中率"] = agg["複勝的中率"] * 100
#     return agg

# def extract_buy_conditions_win(
#     cells: pd.DataFrame,
#     min_roi: float = 110.0,
#     min_n: int = 10
# ) -> pd.DataFrame:
#     out = cells[
#         (cells["単勝ROI"] >= min_roi) &
#         (cells["件数"] >= min_n)
#     ].copy()

#     # Lv → 上昇値 → 人気乖離 → ROI の順で見やすく
#     out = out.sort_values(
#         ["レースレベル", "単勝ROI", "件数"],
#         ascending=[True, False, False]
#     )

#     return out

# def extract_buy_conditions_place(
#     cells: pd.DataFrame,
#     min_roi: float = 105.0,
#     min_n: int = 10
# ) -> pd.DataFrame:
#     out = cells[
#         (cells["複勝ROI"] >= min_roi) &
#         (cells["件数"] >= min_n)
#     ].copy()

#     # Lv → 上昇値 → 人気乖離 → ROI の順で見やすく
#     out = out.sort_values(
#         ["レースレベル", "複勝ROI", "件数"],
#         ascending=[True, False, False]
#     )

#     return out

st.divider()
st.header("■ Lv別・自動抽出された買い条件")

cells = build_condition_cells(df_hm)

# ---- 単勝条件 ----
st.subheader("✅ 単勝・買い条件")
buy_win = extract_buy_conditions_win(
    cells,
    min_roi=win_min_roi,
    min_n=win_min_n
)

if buy_win.empty:
    st.info("単勝の買い条件は見つかりませんでした。")
else:
    buy_win_disp = buy_win.copy()
    buy_win_disp["出力"] = False

    edited_win = st.data_editor(
        buy_win_disp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "出力": st.column_config.CheckboxColumn("CSV出力"),
        },
        disabled=[
            c for c in buy_win_disp.columns if c != "出力"
        ],
    )

# ---- 複勝条件 ----
st.subheader("✅ 複勝・買い条件")
buy_place = extract_buy_conditions_place(
    cells,
    min_roi=plc_min_roi,
    min_n=plc_min_n
)

if buy_place.empty:
    st.info("複勝の買い条件は見つかりませんでした。")
else:
    buy_place_disp = buy_place.copy()
    buy_place_disp["出力"] = False

    edited_plc = st.data_editor(
        buy_place_disp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "出力": st.column_config.CheckboxColumn("CSV出力"),
        },
        disabled=[
            c for c in buy_place_disp.columns if c != "出力"
        ],
    )

    st.caption(
        f"※ 単勝条件：ROI ≧ {win_min_roi:.0f}% / 件数 ≧ {win_min_n}、"
        f"複勝条件：ROI ≧ {plc_min_roi:.0f}% / 件数 ≧ {plc_min_n}"
    )

st.divider()
# st.subheader("📤 買い条件CSVの手動生成")

# if st.button("✅ 選択した条件をCSVに出力"):
# 単勝
win_sel = edited_win[edited_win["出力"]].drop(columns="出力", errors="ignore") \
    if "edited_win" in locals() else pd.DataFrame()
# 複勝
plc_sel = edited_plc[edited_plc["出力"]].drop(columns="出力", errors="ignore") \
    if "edited_plc" in locals() else pd.DataFrame()

if win_sel.empty and plc_sel.empty:
    st.warning("CSVに出力する条件が選択されていません。")
else:
    up_bounds = make_bin_bounds(bins_up, labels_up, "上昇値区分")
    gap_bounds = make_bin_bounds(bins_gap, labels_gap, "人気乖離区分")

    # 単勝CSV
    if not win_sel.empty:
        out_win = (
            win_sel.merge(up_bounds, on="上昇値区分", how="left")
            .merge(gap_bounds, on="人気乖離区分", how="left", suffixes=("_up", "_gap"))
            .rename(columns={
                "low_up": "up_low", "high_up": "up_high", "include_lowest_up": "up_include_lowest",
                "low_gap": "gap_low", "high_gap": "gap_high", "include_lowest_gap": "gap_include_lowest",
            })
        )
        out_win["母集団"] = population_mode
        # --- 前提フィルタ情報（④×⑤）を保存
        out_win["前提_合格数区分"] = "3/3（万全）" if use_comp33 else "ALL"
        out_win["前提_混戦度指標"] = "cv"
        out_win["前提_混戦度上位率"] = top_rate
        out_win["前提_cv閾値"] = cv_threshold if cv_threshold is not None else np.nan
    
    # 複勝CSV
    if not plc_sel.empty:
        out_plc = (
            plc_sel.merge(up_bounds, on="上昇値区分", how="left")
            .merge(gap_bounds, on="人気乖離区分", how="left", suffixes=("_up", "_gap"))
            .rename(columns={
                "low_up": "up_low", "high_up": "up_high", "include_lowest_up": "up_include_lowest",
                "low_gap": "gap_low", "high_gap": "gap_high", "include_lowest_gap": "gap_include_lowest",
            })
        )
        out_plc["母集団"] = population_mode
        # --- 前提フィルタ情報（④×⑤）を保存
        out_plc["前提_合格数区分"] = "3/3（万全）" if use_comp33 else "ALL"
        out_plc["前提_混戦度指標"] = "cv"
        out_plc["前提_混戦度上位率"] = top_rate
        out_plc["前提_cv閾値"] = cv_threshold if cv_threshold is not None else np.nan
        
st.subheader("📤 買い条件CSVの手動生成（ダウンロード）")
if not win_sel.empty:
    csv_bytes_win = df_to_csv_bytes(out_win)
    if st.download_button(
        label="⬇️ 単勝条件CSVをダウンロード",
        data=csv_bytes_win,
        file_name="buy_conditions_full_win.csv",
        mime="text/csv",
    ):
        st.success("✅ 選択した単勝買い条件をCSVに出力しました")

if not plc_sel.empty:
    csv_bytes_plc = df_to_csv_bytes(out_plc)
    if st.download_button(
        label="⬇️ 複勝条件CSVをダウンロード",
        data=csv_bytes_plc,
        file_name="buy_conditions_full_place.csv",
        mime="text/csv",
    ):
        st.success("✅ 選択した複勝買い条件をCSVに出力しました")

# st.success("✅ 選択した買い条件をCSVに出力しました")
# st.caption(f"出力先: {BUY_WIN_FULL_PATH.name}, {BUY_PLC_FULL_PATH.name}")

# =========================================================
# デバッグ表示
# =========================================================
st.write(
    "前走総合利益度 notna:",
    df_filtered["前走総合利益度"].notna().sum(),
    "/",
    len(df_filtered),
)
st.write(
    "利益度上昇値 notna:",
    df_filtered["利益度上昇値"].notna().sum(),
    "/",
    len(df_filtered),
)

st.dataframe(df_filtered.head(50), use_container_width=True)