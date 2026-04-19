import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import re
import io
from datetime import datetime

from core.features import add_component_pass_count, add_race_cv, add_race_deviation_scores, add_deviation_component_pass
from core.history import build_index_history, attach_prev_total
from core.loaders import load_preprocessed, load_return
from core.aggregations import (
    calc_roi_table,
    make_heatmap_table,
    build_condition_cells,
    extract_buy_conditions_win,
    extract_buy_conditions_place,
    add_stability_flags,
    calc_roi_pivot_2d,
)

# =========================================================
# 設定
# =========================================================
DATA_DIR = Path("prof_result")
PREP_DIR = Path("data")
MERGED_RETURN_PATH = Path("./data/return_data_merged.csv")
BUY_WIN_FULL_PATH = Path("./data/buy_conditions_full_win.csv")
BUY_PLC_FULL_PATH = Path("./data/buy_conditions_full_place.csv")


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

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["単勝的中"] = pd.to_numeric(out.get("単勝"), errors="coerce") > 0
    out["複勝的中"] = pd.to_numeric(out.get("複勝"), errors="coerce") > 0

    if {"人気", "総合利益度順位"}.issubset(out.columns):
        out["人気乖離"] = pd.to_numeric(out["人気"], errors="coerce") - pd.to_numeric(out["総合利益度順位"], errors="coerce")
    else:
        out["人気乖離"] = np.nan

    if {"推定人気", "総合利益度順位"}.issubset(out.columns):
        out["推定人気乖離"] = pd.to_numeric(out["推定人気"], errors="coerce") - pd.to_numeric(out["総合利益度順位"], errors="coerce")
    else:
        out["推定人気乖離"] = np.nan

    return out

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

    # def _agg(g: pd.DataFrame) -> pd.Series:
    #     vals = g["総合利益度"].sort_values(ascending=False).to_numpy()
    #     n = len(vals)
    #     top1 = float(vals[0]) if n >= 1 else np.nan
    #     top2 = float(vals[1]) if n >= 2 else np.nan
    #     gap12 = (top1 - top2) if pd.notna(top1) and pd.notna(top2) else np.nan
    #     mean = float(np.nanmean(vals)) if n else np.nan
    #     std = float(np.nanstd(vals, ddof=0)) if n >= 2 else 0.0
    #     cv = float(std / mean) if pd.notna(mean) and mean != 0 else np.nan
    #     return pd.Series({
    #         "n": n,
    #         "mean": mean,
    #         "std": std,
    #         "cv": cv,
    #         "top1": top1,
    #         "top2": top2,
    #         "gap12": gap12,
    #     })

    # out = x.groupby(key, observed=True).apply(_agg).reset_index()

    grp = x.groupby(key, observed=True)["総合利益度"]

    out = grp.agg(n="count", mean="mean", std="std").reset_index()
    # stdはddof=1なのでddof=0に補正
    out["std"] = out["std"] * np.sqrt((out["n"] - 1) / out["n"])
    out.loc[out["n"] < 2, "std"] = 0.0
    out["cv"] = out["std"] / out["mean"].replace(0, np.nan)

    # top1, top2 は nlargest で一括取得
    top2 = grp.nlargest(2).droplevel(-1).groupby(level=[0, 1, 2])
    top1 = top2.nth(0).rename("top1")
    top2_val = top2.nth(1).rename("top2")

    out = out.merge(top1.reset_index(), on=key, how="left")
    out = out.merge(top2_val.reset_index(), on=key, how="left")
    out["gap12"] = out["top1"] - out["top2"]
    
    return out


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

ret_by_date = {}
if df_ret is not None:
    for d_key, grp in df_ret.groupby("開催日"):
        ret_by_date[d_key] = grp

for d, p in dated:
    if d not in target_dates:
        continue

    dfp = load_prof(p)
    dfp["開催日"] = d

    if d in ret_by_date:
        dfp = dfp.merge(
            ret_by_date[d],
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

# 偏差値情報の付与
df = add_race_deviation_scores(df)
df = add_deviation_component_pass(df, threshold=60)


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
    use_deviation_23plus = st.checkbox("偏差値合格数2/3+のみ表示", value=False)
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

if use_comp33 and "合格数区分" in df_hm.columns:
    df_hm = df_hm[df_hm["合格数区分"] == "3/3（万全）"]
if use_deviation_23plus and "偏差値合格数区分" in df_hm.columns:
    df_hm = df_hm[df_hm["偏差値合格数区分"] == "2/3+"]

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

bins_gap = [-999, -5, -1, 3, 7, 999]
labels_gap = ["<=-5", "-4〜-1", "0〜3", "4〜7", "8+"]

st.header("② 人気乖離（確定人気−総合利益度順位）× 回収率・的中率")
if df_hm["人気乖離"].notna().sum() == 0:
    st.info("人気乖離が算出できるデータがありません（人気 or 総合利益度順位が不足）。")
else:
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

st.header("② 推定人気乖離（推定人気−総合利益度順位）× 回収率・的中率")
if df_hm["推定人気乖離"].notna().sum() == 0:
    st.info("推定人気乖離が算出できるデータがありません（推定人気データが不足）。")
else:
    df_hm["推定人気乖離区分"] = pd.cut(
        df_hm["推定人気乖離"], bins=bins_gap, labels=labels_gap, include_lowest=True
    )
    t2_est = calc_roi_table(df_hm.dropna(subset=["推定人気乖離区分"]), "推定人気乖離区分")
    st.dataframe(
        t2_est.style.format({
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

if tab == "単勝回収率":
    st.dataframe(pivot_win_index.style.format("{:.1f}%"), use_container_width=True)
elif tab == "複勝回収率":
    st.dataframe(pivot_plc_index.style.format("{:.1f}%"), use_container_width=True)
else:
    st.dataframe(pivot_n_index.style.format("{:.0f}"), use_container_width=True)

# この後のヒートマップ・買い条件描画は区分が両方揃っている行のみを対象
df_hm_pre = df_hm.copy()  # dropna 前のスナップショット（推定用に使う）
df_hm = df_hm.dropna(subset=["上昇値区分", "人気乖離区分"])
# 推定人気乖離区分 列が未生成の場合（推定データなし）は空DFにする
if "推定人気乖離区分" in df_hm_pre.columns:
    df_hm_est = df_hm_pre.dropna(subset=["上昇値区分", "推定人気乖離区分"])
else:
    df_hm_est = df_hm_pre.iloc[0:0].copy()


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


st.header("⑥-1 偏差値合格数 × 混戦度（cv）")

need_cols = ["偏差値合格数区分", "混戦度区分"]
if any(c not in df_hm.columns for c in need_cols):
    st.info("偏差値合格数区分 または 混戦度区分 が存在しないため、⑥-1 を表示できません。")
else:
    piv = calc_roi_pivot_2d(df_hm, row_col="偏差値合格数区分", col_col="混戦度区分")

    if not piv:
        st.info("⑥-1 の集計に必要なデータが不足しています（欠損など）。")
    else:
        tabA = st.radio(
            "表示指標（⑥-1）",
            ["頭数", "単勝的中率", "複勝的中率", "単勝回収率", "複勝回収率"],
            horizontal=True,
            key="tab_stepA"
        )
        fmt = "{:.0f}" if tabA == "頭数" else "{:.1f}%"
        st.dataframe(
            piv[tabA].style.format(fmt),
            use_container_width=True
        )
        st.caption("※ 行=偏差値合格数区分、列=cvの混戦度区分（qcut）、はずれは0円で回収率を算出")


st.header("⑥-2 偏差値合格数 × 利益度上昇値")

need_cols = ["偏差値合格数区分", "上昇値区分"]
if any(c not in df_hm.columns for c in need_cols):
    st.info("偏差値合格数区分 または 上昇値区分 が存在しないため、⑥-2 を表示できません。")
else:
    piv = calc_roi_pivot_2d(df_hm, row_col="偏差値合格数区分", col_col="上昇値区分")

    if not piv:
        st.info("⑥-2 の集計に必要なデータが不足しています（欠損など）。")
    else:
        tabB = st.radio(
            "表示指標（⑥-2）",
            ["頭数", "単勝的中率", "複勝的中率", "単勝回収率", "複勝回収率"],
            horizontal=True,
            key="tab_stepB"
        )
        fmt = "{:.0f}" if tabB == "頭数" else "{:.1f}%"
        st.dataframe(
            piv[tabB].style.format(fmt),
            use_container_width=True
        )
        st.caption("※ 行=偏差値合格数区分、列=利益度上昇値区分")
        

st.header("⑦ ヒートマップ：Lv × 上昇値区分 × 人気乖離区分")

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
        # hm_n を data と同じ shape にそろえる
        n_aligned = hm_n.reindex_like(data).fillna(0)

        # numpy で一括判定
        vals = data.values.astype(float)
        ns = n_aligned.values.astype(float)
        styles_arr = np.empty(vals.shape, dtype=object)

        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                styles_arr[i, j] = colorize(vals[i, j], ns[i, j])

        return pd.DataFrame(styles_arr, index=data.index, columns=data.columns)

    if metric == "件数":
        sty = hm.style.apply(apply_styles, axis=None).format("{:.0f}")
    else:
        sty = hm.style.apply(apply_styles, axis=None).format("{:.1f}")

    st.caption("※ サンプル数が少ないセル（件数<5）は薄く表示します。")
    st.dataframe(sty, use_container_width=True, height=480)


st.divider()
st.header("⑦ 推定人気乖離ヒートマップ：Lv × 上昇値区分 × 推定人気乖離区分")

metric_est = st.radio(
    "表示する指標（推定）",
    ["単勝ROI", "複勝ROI", "単勝的中率", "複勝的中率", "件数"],
    horizontal=True,
    key="metric_est",
)

hm_est = make_heatmap_table(df_hm_est, metric_est, gap_col="推定人気乖離区分")

if hm_est.empty:
    st.info("推定ヒートマップ表示に必要なデータが不足しています。")
else:
    hm_est_n = make_heatmap_table(df_hm_est, "件数", gap_col="推定人気乖離区分")

    def apply_styles_est(data):
        n_aligned = hm_est_n.reindex_like(data).fillna(0)
        vals = data.values.astype(float)
        ns = n_aligned.values.astype(float)
        styles_arr = np.empty(vals.shape, dtype=object)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                styles_arr[i, j] = colorize(vals[i, j], ns[i, j])
        return pd.DataFrame(styles_arr, index=data.index, columns=data.columns)

    if metric_est == "件数":
        sty_est = hm_est.style.apply(apply_styles_est, axis=None).format("{:.0f}")
    else:
        sty_est = hm_est.style.apply(apply_styles_est, axis=None).format("{:.1f}")

    st.caption("※ サンプル数が少ないセル（件数<5）は薄く表示します。")
    st.dataframe(sty_est, use_container_width=True, height=480)


st.divider()
st.header("■ Lv別・自動抽出された買い条件（確定人気ベース）")

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
    buy_win = add_stability_flags(
        cond_df=buy_win,
        df_all=df_hm,
        bet_type="win",
        min_roi=float(win_min_roi),
        min_n=int(win_min_n),
        date_col="開催日",
    )
    buy_win_disp = buy_win.copy()
    buy_win_disp["出力"] = False

    edited_win = st.data_editor(
        buy_win_disp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "出力": st.column_config.CheckboxColumn("CSV出力"),
            "期間分割OK": st.column_config.CheckboxColumn("6:4検証OK"),
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
    # 期間分割検証（6:4）を付与
    # ※ df_hm はすでに前提フィルタ（3/3 & cv上位）などがかかった母集団
    buy_place = add_stability_flags(
        cond_df=buy_place,
        df_all=df_hm,
        bet_type="place",
        min_roi=float(plc_min_roi),
        min_n=int(plc_min_n),
        date_col="開催日",
    )
    buy_place_disp = buy_place.copy()
    buy_place_disp["出力"] = False

    edited_plc = st.data_editor(
        buy_place_disp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "出力": st.column_config.CheckboxColumn("CSV出力"),
            "期間分割OK": st.column_config.CheckboxColumn("6:4検証OK"),
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


# ==========================================================
# ■ 推定人気ベースの買い条件
# ==========================================================
st.divider()
st.header("■ Lv別・自動抽出された買い条件（推定人気ベース）")

# 6:4検証は推定人気乖離区分で評価するため、df_hm_est の列名を一時的に合わせる
df_hm_est_for_stability = df_hm_est.copy()
if "推定人気乖離区分" in df_hm_est_for_stability.columns:
    df_hm_est_for_stability["人気乖離区分"] = df_hm_est_for_stability["推定人気乖離区分"]

cells_est = build_condition_cells(df_hm_est, gap_col="推定人気乖離区分")

# ---- 単勝条件（推定） ----
st.subheader("✅ 単勝・買い条件（推定人気）")
buy_win_est = extract_buy_conditions_win(cells_est, min_roi=win_min_roi, min_n=win_min_n)

if buy_win_est.empty:
    st.info("単勝の買い条件（推定人気）は見つかりませんでした。")
else:
    buy_win_est = add_stability_flags(
        cond_df=buy_win_est,
        df_all=df_hm_est_for_stability,
        bet_type="win",
        min_roi=float(win_min_roi),
        min_n=int(win_min_n),
        date_col="開催日",
    )
    buy_win_est_disp = buy_win_est.copy()
    buy_win_est_disp["出力"] = False

    edited_win_est = st.data_editor(
        buy_win_est_disp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "出力": st.column_config.CheckboxColumn("CSV出力"),
            "期間分割OK": st.column_config.CheckboxColumn("6:4検証OK"),
        },
        disabled=[c for c in buy_win_est_disp.columns if c != "出力"],
    )

# ---- 複勝条件（推定） ----
st.subheader("✅ 複勝・買い条件（推定人気）")
buy_place_est = extract_buy_conditions_place(cells_est, min_roi=plc_min_roi, min_n=plc_min_n)

if buy_place_est.empty:
    st.info("複勝の買い条件（推定人気）は見つかりませんでした。")
else:
    buy_place_est = add_stability_flags(
        cond_df=buy_place_est,
        df_all=df_hm_est_for_stability,
        bet_type="place",
        min_roi=float(plc_min_roi),
        min_n=int(plc_min_n),
        date_col="開催日",
    )
    buy_place_est_disp = buy_place_est.copy()
    buy_place_est_disp["出力"] = False

    edited_plc_est = st.data_editor(
        buy_place_est_disp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "出力": st.column_config.CheckboxColumn("CSV出力"),
            "期間分割OK": st.column_config.CheckboxColumn("6:4検証OK"),
        },
        disabled=[c for c in buy_place_est_disp.columns if c != "出力"],
    )

    st.caption(
        f"※ 単勝条件：ROI ≧ {win_min_roi:.0f}% / 件数 ≧ {win_min_n}、"
        f"複勝条件：ROI ≧ {plc_min_roi:.0f}% / 件数 ≧ {plc_min_n}"
    )

st.divider()

# CSV 出力（推定人気ベース）
win_sel_est = edited_win_est[edited_win_est["出力"]].drop(columns="出力", errors="ignore") \
    if "edited_win_est" in locals() else pd.DataFrame()
plc_sel_est = edited_plc_est[edited_plc_est["出力"]].drop(columns="出力", errors="ignore") \
    if "edited_plc_est" in locals() else pd.DataFrame()

if win_sel_est.empty and plc_sel_est.empty:
    st.warning("推定人気ベースの条件：CSVに出力する条件が選択されていません。")
else:
    up_bounds = make_bin_bounds(bins_up, labels_up, "上昇値区分")
    gap_bounds = make_bin_bounds(bins_gap, labels_gap, "人気乖離区分")

    if not win_sel_est.empty:
        out_win_est = (
            win_sel_est.merge(up_bounds, on="上昇値区分", how="left")
            .merge(gap_bounds, on="人気乖離区分", how="left", suffixes=("_up", "_gap"))
            .rename(columns={
                "low_up": "up_low", "high_up": "up_high", "include_lowest_up": "up_include_lowest",
                "low_gap": "gap_low", "high_gap": "gap_high", "include_lowest_gap": "gap_include_lowest",
            })
        )
        out_win_est["母集団"] = population_mode
        out_win_est["前提_合格数区分"] = "3/3（万全）" if use_comp33 else "ALL"
        out_win_est["前提_混戦度指標"] = "cv"
        out_win_est["前提_混戦度上位率"] = top_rate
        out_win_est["前提_cv閾値"] = cv_threshold if cv_threshold is not None else np.nan

    if not plc_sel_est.empty:
        out_plc_est = (
            plc_sel_est.merge(up_bounds, on="上昇値区分", how="left")
            .merge(gap_bounds, on="人気乖離区分", how="left", suffixes=("_up", "_gap"))
            .rename(columns={
                "low_up": "up_low", "high_up": "up_high", "include_lowest_up": "up_include_lowest",
                "low_gap": "gap_low", "high_gap": "gap_high", "include_lowest_gap": "gap_include_lowest",
            })
        )
        out_plc_est["母集団"] = population_mode
        out_plc_est["前提_合格数区分"] = "3/3（万全）" if use_comp33 else "ALL"
        out_plc_est["前提_混戦度指標"] = "cv"
        out_plc_est["前提_混戦度上位率"] = top_rate
        out_plc_est["前提_cv閾値"] = cv_threshold if cv_threshold is not None else np.nan

st.subheader("📤 買い条件CSV（推定人気ベース）の手動生成（ダウンロード）")
if not win_sel_est.empty:
    csv_bytes_win_est = df_to_csv_bytes(out_win_est)
    if st.download_button(
        label="⬇️ 単勝条件CSV（推定）をダウンロード",
        data=csv_bytes_win_est,
        file_name="buy_conditions_full_win_推定.csv",
        mime="text/csv",
        key="dl_win_est",
    ):
        st.success("✅ 選択した単勝買い条件（推定）をCSVに出力しました")

if not plc_sel_est.empty:
    csv_bytes_plc_est = df_to_csv_bytes(out_plc_est)
    if st.download_button(
        label="⬇️ 複勝条件CSV（推定）をダウンロード",
        data=csv_bytes_plc_est,
        file_name="buy_conditions_full_place_推定.csv",
        mime="text/csv",
        key="dl_plc_est",
    ):
        st.success("✅ 選択した複勝買い条件（推定）をCSVに出力しました")


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