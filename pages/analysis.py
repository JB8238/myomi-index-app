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
st.caption("prof_result（指数） + return_data（結果/払戻）を統合して横断分析します（着順は不使用）")

st.page_link("app.py", label="← 開催レース一覧へ戻る", icon="🏠", use_container_width=True)
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

def list_csv_files(data_dir: Path):
    return sorted([p for p in data_dir.rglob("*.csv") if p.is_file()])

def pick_latest_by_filename(files):
    dated = []
    for f in files:
        d = extract_yyyymmdd_from_name(f.name)
        if d is not None:
            dated.append((d, f))
    dated.sort()
    return dated[-1][1] if dated else None

def pick_return_data_csv(root: Path, target_date: int) -> Path | None:
    """root/YYYY/return_data_YYYYMMDD.csv を想定して一致ファイルを返す"""
    year_dir = root / str(target_date)[:4]
    if not year_dir.exists():
        return None
    files = sorted(year_dir.glob("return_data_*.csv"))
    for f in files:
        d = extract_yyyymmdd_from_name(f.name)
        if d == target_date:
            return f
    return None

def normalize_place_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\u3000", " ").str.strip()

@st.cache_data(show_spinner="📥 prof_result を読み込んでいます…")
def load_prof(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp932")
    df.columns = [str(c).strip() for c in df.columns]

    # キー整形
    if "場所" in df.columns:
        df["場所"] = normalize_place_series(df["場所"])
    if "R" in df.columns:
        df["R"] = pd.to_numeric(df["R"], errors="coerce")
    if "馬番" in df.columns:
        df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")

    # 数値化（存在するものだけ）
    for c in ["総合利益度", "総合利益度順位", "前走総合利益度"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

@st.cache_data(show_spinner="📥 return_data を読み込んでいます…")
def load_return(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp932")
    df.columns = [str(c).strip() for c in df.columns]
    # 全角Ｒ→R
    df.rename(columns={"Ｒ": "R"}, inplace=True)

    # 「着」は使わないので混線防止で捨てる（存在すれば）
    if "着" in df.columns:
        df.drop(columns=["着"], inplace=True)

    # 必要列だけ残す（キー+結果）
    keep = [c for c in ["場所", "R", "馬番"] + RESULT_COLS if c in df.columns]
    df = df[keep].copy()

    # キー整形
    if "場所" in df.columns:
        df["場所"] = normalize_place_series(df["場所"])
    if "R" in df.columns:
        df["R"] = pd.to_numeric(df["R"], errors="coerce")
    if "馬番" in df.columns:
        df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")

    # 結果列 数値化
    for c in RESULT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 馬番重複があれば先頭（念のため）
    if "馬番" in df.columns:
        df = df.sort_values("馬番").drop_duplicates(subset=["場所", "R", "馬番"], keep="first")

    return df

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 利益度上昇値（analysis側でも計算）
    if {"総合利益度", "前走総合利益度"}.issubset(out.columns):
        cur_int = np.trunc(pd.to_numeric(out["総合利益度"], errors="coerce"))
        prev = pd.to_numeric(out["前走総合利益度"], errors="coerce")
        out["利益度上昇値"] = cur_int - prev
    else:
        out["利益度上昇値"] = np.nan

    # 的中フラグ（着順は使わない）
    if "単勝" in out.columns:
        out["単勝的中"] = pd.to_numeric(out["単勝"], errors="coerce") > 0
    else:
        out["単勝的中"] = False

    if "複勝" in out.columns:
        out["複勝的中"] = pd.to_numeric(out["複勝"], errors="coerce") > 0
    else:
        out["複勝的中"] = False

    # 人気乖離（人気は return_data 側）
    if {"人気", "総合利益度順位"}.issubset(out.columns):
        out["人気乖離"] = out["人気"] - out["総合利益度順位"]
    else:
        out["人気乖離"] = np.nan

    return out

def summarize_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """存在保証のある列だけで集計→回収率は後付け（安全）"""
    g = (
        df.groupby(group_col, observed=True)
        .agg(
            頭数=("馬名", "count") if "馬名" in df.columns else ("馬番", "count"),
            単勝的中率=("単勝的中", "mean") if "単勝的中" in df.columns else ("馬番", lambda x: np.nan),
            複勝的中率=("複勝的中", "mean") if "複勝的中" in df.columns else ("馬番", lambda x: np.nan),
        )
        .reset_index()
    )

    # 回収率（return_data統合済みなら単勝/複勝列がある）
    # 注意：単勝/複勝は「100円あたりの払戻（円）」なので、平均値がそのまま回収率(%)相当
    if "単勝" in df.columns:
        tmp = df.groupby(group_col, observed=True)["単勝"].mean().reset_index(name="単勝回収率")
        g = g.merge(tmp, on=group_col, how="left")
    else:
        g["単勝回収率"] = np.nan

    if "複勝" in df.columns:
        tmp = df.groupby(group_col, observed=True)["複勝"].mean().reset_index(name="複勝回収率")
        g = g.merge(tmp, on=group_col, how="left")
    else:
        g["複勝回収率"] = np.nan

    # %表示用に的中率のみ×100（回収率は既に“%相当”の値）
    if "単勝的中率" in g.columns:
        g["単勝的中率"] = g["単勝的中率"] * 100
    if "複勝的中率" in g.columns:
        g["複勝的中率"] = g["複勝的中率"] * 100

    return g

@st.cache_data(show_spinner="📚 prof_result 履歴（前走参照用）を整備しています…")
def build_prof_history(data_dir: Path) -> pd.DataFrame:
    files = sorted([p for p in data_dir.rglob("*.csv") if p.is_file()])
    rows = []

    for p in files:
        d = extract_yyyymmdd_from_name(p.name)
        if d is None:
            continue

        mtime = p.stat().st_mtime

        try:
            df0 = pd.read_csv(p, encoding="cp932")
        except Exception:
            df0 = pd.read_csv(p, encoding="cp932", errors="ignore")

        # 必須列がないものはスキップ
        if "馬名" not in df0.columns or "総合利益度" not in df0.columns:
            continue

        df0.columns = [str(c).strip() for c in df0.columns]

        keep = [c for c in ["馬名", "総合利益度"] if c in df0.columns]
        df1 = df0[keep].copy()

        df1["馬名"] = df1["馬名"].astype(str).str.strip()
        df1["総合利益度"] = pd.to_numeric(df1["総合利益度"], errors="coerce")

        df1["__date"] = int(d)
        df1["__mtime"] = float(mtime)
        df1["__file"] = p.name

        rows.append(df1)

    if not rows:
        return pd.DataFrame(columns=["馬名", "総合利益度", "__date", "__mtime", "__file"])

    hist = pd.concat(rows, ignore_index=True)
    hist = hist[hist["馬名"].notna() & (hist["馬名"].astype(str).str.strip() != "")]
    return hist

# -----------------------------------
# CSV選択（query_params → session_state → latest）
# -----------------------------------
qp = st.query_params
csv_name = qp.get("csv")
files = list_csv_files(DATA_DIR)
selected_file = None
if csv_name:
    for p in files:
        if p.name == csv_name:
            selected_file = p
            break
if selected_file is None:
    selected_file = st.session_state.get("selected_prof_csv")
if selected_file is None:
    selected_file = pick_latest_by_filename(files)

# =========================================================
# データ読み込み（prof_result）＋開催日選択
# =========================================================
files = list_csv_files(DATA_DIR)
if not files:
    st.error("prof_result にCSVが見つかりません。")
    st.stop()

dated_files = [(extract_yyyymmdd_from_name(p.name), p) for p in files]
dated_files = [(d, p) for d, p in dated_files if d is not None]
dated_files.sort(key=lambda x: x[0])

available_dates = [d for d, _ in dated_files]
latest_date = available_dates[-1]
latest_prof_path = dict(dated_files)[latest_date]

selected_file = st.session_state.get("selected_prof_csv")

if selected_file is not None:
    prof_path = selected_file
else:
    prof_path = pick_latest_by_filename(list_csv_files(DATA_DIR))

with st.sidebar:
    st.header("分析対象")
    mode = st.radio("対象期間", ["最新日だけ", "日付を選ぶ", "全期間（全CSV統合）"], index=0)
    if mode == "最新日だけ":
        target_dates = [latest_date]
    elif mode == "日付を選ぶ":
        # 複数選択可
        target_dates = st.multiselect("開催日（YYYYMMDD）", options=available_dates, default=[latest_date])
        if not target_dates:
            target_dates = [latest_date]
    else:
        target_dates = available_dates

    only_pass = st.checkbox("合格馬のみ（総合利益度>=0）", value=False)

st.caption(f"return_data 参照ルート: {str(MERGED_RETURN_PATH)}")

# =========================================================
# 統合データセット構築
# =========================================================
frames = []
missing_return = []

for d in target_dates:
    prof_path = dict(dated_files).get(d)
    if prof_path is None:
        continue

    df_prof = load_prof(selected_file)
    df_prof["開催日"] = d

    if MERGED_RETURN_PATH.exists():
        df_ret = pd.read_csv(MERGED_RETURN_PATH, encoding="utf-8-sig")
        df_ret = df_ret[df_ret["開催日"] == d]
    else:
        df_ret = None
    
    if df_ret is not None and not df_ret.empty:
        df_all = df_prof.merge(
            df_ret,
            on=["場所", "R", "馬番"],
            how="left",
            validate="m:1"
        )
    else:
        df_all = df_prof

    df_all = add_derived_columns(df_all)

    # 合格馬のみフィルタ（任意）
    if only_pass and "総合利益度" in df_all.columns:
        df_all = df_all[df_all["総合利益度"].notna() & (df_all["総合利益度"] >= 0)]

    frames.append(df_all)

if not frames:
    st.error("分析対象データが空です。")
    st.stop()

df = pd.concat(frames, ignore_index=True)

# ======================================
# 前走総合利益度（prof_result履歴から付与）
# ======================================
history = build_prof_history(DATA_DIR)

def find_prev_total(horse_name: str, cur_date: int, cur_file: str) -> float:
    if horse_name is None or pd.isna(horse_name):
        return np.nan
    hname = str(horse_name).strip()
    if hname == "":
        return np.nan

    h = history[(history["馬名"] == hname) & (history["__file"] != cur_file)]
    if h.empty:
        return np.nan

    # “直近過去”を選ぶ（date優先、同日ならmtime）
    h = h[h["__date"] < cur_date]
    if h.empty:
        return np.nan

    h = h.sort_values(["__date", "__mtime"])
    v = h.iloc[-1]["総合利益度"]
    return float(v) if pd.notna(v) else np.nan

# df に必要な情報がある場合だけ
if "開催日" in df.columns and "馬名" in df.columns:
    # どのファイル（当日CSV）から来た行か判定するため、最低限 cur_file を作る
    # 今回は開催日から "prof_result_YYYYMMDD.csv" を仮想的に作る（あなたの命名規則に合わせて必要なら調整）
    df["__cur_file"] = df["開催日"].astype(str).apply(lambda x: f"prof_result_{x}.csv")

    df["前走総合利益度"] = df.apply(
        lambda r: find_prev_total(r["馬名"], int(r["開催日"]), r["__cur_file"]),
        axis=1
    )

    # 利益度上昇値を再計算（上書き）
    if "総合利益度" in df.columns:
        cur_int = np.trunc(pd.to_numeric(df["総合利益度"], errors="coerce"))
        prev = pd.to_numeric(df["前走総合利益度"], errors="coerce")
        df["利益度上昇値"] = cur_int - prev

# return_data不足日がある場合は表示（分析は継続）
if missing_return:
    st.warning(f"return_data が見つからない開催日がありました（回収率がNaNになります）: {missing_return[:10]}{'…' if len(missing_return) > 10 else ''}")

st.subheader("データ概要")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("対象開催日数", len(set(target_dates)))
with c2:
    st.metric("対象行数（馬数）", len(df))
with c3:
    st.metric("return_data 欠損日数", len(missing_return))

st.divider()

# =========================================================
# 分析①：利益度上昇値 × 回収率・的中率（アイデア1）
# =========================================================
st.header("① 利益度上昇値 × 回収率・的中率")

if df["利益度上昇値"].notna().sum() == 0:
    st.info("利益度上昇値が計算できるデータ（総合利益度＆前走総合利益度）が不足しています。")
else:
    bins_up = [-999, 0, 5, 10, 999]
    labels_up = ["<=0", "1–5", "6–10", "11+"]

    df["上昇値区分"] = pd.cut(df["利益度上昇値"], bins=bins_up, labels=labels_up)

    summary_up = summarize_by_group(df.dropna(subset=["上昇値区分"]), "上昇値区分")

    st.dataframe(
        summary_up.style.format({
            "単勝的中率": "{:.1f}%",
            "複勝的中率": "{:.1f}%",
            "単勝回収率": "{:.1f}%",
            "複勝回収率": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# =========================================================
# 分析②：人気乖離 × 回収率・的中率（アイデア4）
# =========================================================
st.header("② 人気乖離（人気−総合利益度順位）× 回収率・的中率")

if df["人気乖離"].notna().sum() == 0:
    st.info("人気乖離が計算できるデータ（人気＆総合利益度順位）が不足しています。")
else:
    bins_gap = [-999, -5, -1, 3, 7, 999]
    labels_gap = ["<=-5", "-4〜-1", "0〜3", "4〜7", "8+"]

    df["人気乖離区分"] = pd.cut(df["人気乖離"], bins=bins_gap, labels=labels_gap)

    summary_gap = summarize_by_group(df.dropna(subset=["人気乖離区分"]), "人気乖離区分")

    st.dataframe(
        summary_gap.style.format({
            "単勝的中率": "{:.1f}%",
            "複勝的中率": "{:.1f}%",
            "単勝回収率": "{:.1f}%",
            "複勝回収率": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )