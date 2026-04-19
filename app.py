import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
import hmac

from buy_condition_logic import load_buy_conditions, apply_buy_conditions, race_badge_from_horses
from core.features import add_component_pass_count, add_race_cv_local, add_race_deviation_scores, add_deviation_component_pass
from core.history import find_prev_total, build_prof_history


DATA_DIR = Path("prof_result")
PREP_DIR = Path("data")
BUY_WIN_FULL_PATH = Path("./data/buy_conditions_full_win.csv")
BUY_PLC_FULL_PATH = Path("./data/buy_conditions_full_place.csv")
BUY_WIN_EST_PATH  = Path("./data/buy_conditions_full_win_推定.csv")
BUY_PLC_EST_PATH  = Path("./data/buy_conditions_full_place_推定.csv")
MERGED_RETURN_PATH = Path("./data/return_data_merged.csv")
SMARTRC_DIR = Path("./data/smartrc")

# パスワード認証
def check_password():
    def password_entered():
        if hmac.compare_digest(
            st.session_state["password"], st.secrets["password"]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("パスワード", type="password", key="password", on_change=password_entered)
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("パスワードが正しくありません")
    return False

if not check_password():
    st.stop()


st.set_page_config(
    page_title="開催レース一覧",
    page_icon="🏇",
    layout="wide",
)

if "selected_prof_csv" not in st.session_state:
    st.session_state["selected_prof_csv"] = None

st.title("🏇 開催レース一覧")

# -----------------------------------
# ユーティリティ
# -----------------------------------
def extract_yyyymmdd_from_name(filename: str):
    m = re.findall(r"(\d{8})", filename)
    for s in reversed(m):
        try:
            datetime.strptime(s, "%Y%m%d")
            return s
        except ValueError:
            pass
    return None

def list_csv_files():
    return sorted([p for p in DATA_DIR.rglob("*.csv") if p.is_file()])

def pick_latest_by_filename(files):
    dated = []
    for f in files:
        d = extract_yyyymmdd_from_name(f.name)
        if d:
            dated.append((d, f))
    dated.sort()
    return dated[-1][1] if dated else None

@st.cache_data(show_spinner="📘 レースレベル (preprocessed_data) を読み込んでいます…")
def load_race_level_map(prep_root: Path, target_date: str | None) -> dict:
    """
    data/YYYY/YYYYMMDD/preprocessed_data_*.csv から
    (場所, R) -> レースレベル の辞書を作る
    """
    if target_date is None:
        return {}
    
    year = target_date[:4]
    ymd_dir = prep_root / year / target_date
    if not ymd_dir.exists():
        return {}
    
    files = list(ymd_dir.glob("preprocessed_data_*.csv"))
    if not files:
        return {}
    
    # 同日内で複数あれば名前順で最後（安定）
    path = sorted(files)[-1]

    dfp = pd.read_csv(path, encoding="utf-8")

    required = {"場所", "R", "レースレベル"}
    if not required.issubset(dfp.columns):
        return {}
    
    dfp = dfp.copy()
    dfp["R"] = pd.to_numeric(dfp["R"], errors="coerce")
    dfp["レースレベル"] = (
        dfp["レースレベル"]
        .astype(str)
        .str.strip()    # ← Lv3 などの末尾空白対策
    )

    level_map = {}
    for (place, r), g in dfp.dropna(subset=["R"]).groupby(["場所", "R"]):
        lv = g["レースレベル"].dropna()
        if not lv.empty:
            level_map[(place, int(r))] = lv.mode().iloc[0]
    
    return level_map


# -----------------------------------
# データ読み込み
# -----------------------------------
files = list_csv_files()

# サイドバーで選択されたCSVを使用（なければ最新）
selected_file = st.session_state.get("selected_prof_csv")
if selected_file is None:
    selected_file = pick_latest_by_filename(files)
df = pd.read_csv(selected_file, encoding="cp932")
kaisai_date = extract_yyyymmdd_from_name(selected_file.name)

if MERGED_RETURN_PATH.exists():
    df_return = pd.read_csv(MERGED_RETURN_PATH, encoding="utf-8-sig")
    df_return.columns = [str(c).strip() for c in df_return.columns]
    df_return.rename(columns={"Ｒ": "R"}, inplace=True)

    # 型と表記をそろえる
    for c in ["開催日", "R", "馬番"]:
        if c in df_return.columns:
            df_return[c] = pd.to_numeric(df_return[c], errors="coerce")
    if "場所" in df_return.columns:
        df_return["場所"] = df_return["場所"].astype(str).str.replace("\u3000", " ").str.strip()

    if "場所" in df.columns:
        df["場所"] = df["場所"].astype(str).str.replace("\u3000", " ").str.strip()
    if "R" in df.columns:
        df["R"] = pd.to_numeric(df["R"], errors="coerce")
    if "馬番" in df.columns:
        df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")

    # 当日分だけ JOIN
    df_ret_day = df_return[df_return["開催日"] == int(kaisai_date)]
    df = df.merge(
        df_ret_day,
        on=["場所", "R", "馬番"],
        how="left",
        validate="m:1",
    )

# --- smartrc 推定人気・人気ランク のマージ ---
smartrc_path = SMARTRC_DIR / f"smartrc_{kaisai_date}.csv"
if smartrc_path.exists():
    df_smartrc = pd.read_csv(smartrc_path, encoding="utf-8-sig")
    df_smartrc["場所"] = df_smartrc["場所"].astype(str).str.replace("\u3000", " ").str.strip()
    df_smartrc["R"] = pd.to_numeric(df_smartrc["R"], errors="coerce")
    df_smartrc["馬番"] = pd.to_numeric(df_smartrc["馬番"], errors="coerce")
    df_smartrc["推定人気"] = pd.to_numeric(df_smartrc["推定人気"], errors="coerce")
    df = df.merge(
        df_smartrc[["場所", "R", "馬番", "推定人気", "人気ランク"]],
        on=["場所", "R", "馬番"],
        how="left",
    )

if selected_file is None:
    st.error("prof_result にCSVが見つかりません")
    st.stop()

# -----------------------------------
# ✅ 開催日単位：買い条件あり判定
# -----------------------------------
win_path = BUY_WIN_FULL_PATH
win_mtime = win_path.stat().st_mtime if win_path.exists() else 0.0
cond_win = load_buy_conditions(str(win_path), win_mtime)

plc_path = BUY_PLC_FULL_PATH
plc_mtime = plc_path.stat().st_mtime if plc_path.exists() else 0.0
cond_plc = load_buy_conditions(str(plc_path), plc_mtime)

win_est_mtime = BUY_WIN_EST_PATH.stat().st_mtime if BUY_WIN_EST_PATH.exists() else 0.0
cond_win_est = load_buy_conditions(str(BUY_WIN_EST_PATH), win_est_mtime)
plc_est_mtime = BUY_PLC_EST_PATH.stat().st_mtime if BUY_PLC_EST_PATH.exists() else 0.0
cond_plc_est = load_buy_conditions(str(BUY_PLC_EST_PATH), plc_est_mtime)

csv_pop = None
if "母集団" in cond_win.columns and cond_win["母集団"].dropna().astype(str).str.strip().any():
    csv_pop = cond_win["母集団"].dropna().astype(str).iloc[0]
elif "母集団" in cond_plc.columns and cond_plc["母集団"].dropna().astype(str).str.strip().any():
    csv_pop = cond_plc["母集団"].dropna().astype(str).iloc[0]

if csv_pop:
    st.caption(f"📌 現在参照中の戦略CSV母集団（analysis出力時）: {csv_pop}")

# 利益度上昇値（analysisと同じ定義）
if "総合利益度" in df.columns:
    history = build_prof_history(DATA_DIR)
    cur_date = int(kaisai_date)
    cur_mtime = selected_file.stat().st_mtime
    cur_file = selected_file.name

    df["前走総合利益度"] = df["馬名"].astype(str).apply(
        lambda n: find_prev_total(history, n, cur_date, cur_mtime, cur_file)
    )
    df["利益度上昇値"] = (
        pd.to_numeric(df["総合利益度"], errors="coerce")
        - pd.to_numeric(df["前走総合利益度"], errors="coerce")
    )

# 人気乖離（確定人気のみ）・推定人気乖離（推定人気のみ）
if "総合利益度順位" in df.columns:
    df["総合利益度順位"] = pd.to_numeric(df["総合利益度順位"], errors="coerce")
    if "人気" in df.columns:
        df["人気"] = pd.to_numeric(df["人気"], errors="coerce")
    if "推定人気" in df.columns:
        df["推定人気"] = pd.to_numeric(df["推定人気"], errors="coerce")
    df["人気乖離"] = (
        df["人気"] - df["総合利益度順位"]
        if "人気" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    df["推定人気乖離"] = (
        df["推定人気"] - df["総合利益度順位"]
        if "推定人気" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
else:
    df["人気乖離"] = np.nan
    df["推定人気乖離"] = np.nan

# 偏差値情報の付与
df = add_race_deviation_scores(df)
df = add_deviation_component_pass(df, threshold=60)

# レースレベル（preprocessed_data 由来）
level_map = load_race_level_map(PREP_DIR, kaisai_date)
race_has_condition = {}

for (place, r), g in df.groupby(["場所", "R"]):
    lv = level_map.get((place, int(r)))
    # index_viewと同じ「馬単位判定」を付与してから集約する
    g2 = g.copy()
    # index_viewデフォルト（合格馬>=0）に合わせたいなら下行を有効化（推奨）
    g2 = g2[g2["総合利益度"].notna() & (pd.to_numeric(g2["総合利益度"], errors="coerce") >= 0)]

    # index_view と同じ「合格馬条件」を適用
    g_base = g.copy()
    if "総合利益度" in g_base.columns:
        g_base["総合利益度"] = pd.to_numeric(g_base["総合利益度"], errors="coerce")
        g_base = g_base[g_base["総合利益度"].notna() & (g_base["総合利益度"] >= 0)]

    # 前提条件 (cv & 合格数) のための列を付与
    g_base = add_component_pass_count(g_base)
    g_base = add_race_cv_local(g_base)
    # 確定人気ベースの判定
    g2 = apply_buy_conditions(g_base, lv, cond_win, cond_plc)
    badge = race_badge_from_horses(g2)
    # 推定人気ベースの判定（推定人気専用条件CSVを使用）
    g2_est = apply_buy_conditions(g_base, lv, cond_win_est, cond_plc_est, pop_col="推定人気乖離", suffix="_推定")
    badge_est = race_badge_from_horses(g2_est, win_col="単勝_条件_推定", plc_col="複勝_条件_推定")
    race_has_condition[(place, int(r))] = (badge, badge_est)


if not {"場所", "R"}.issubset(df.columns):
    st.error("CSVに「場所」「R」列が必要です")
    st.stop()

with st.sidebar:
    st.header("データ設定")

    files = list_csv_files()
    if not files:
        st.error("prof_result にCSVが見つかりません。")
        st.stop()

    files_sorted = sorted(
        files,
        key=lambda p: (extract_yyyymmdd_from_name(p.name) or 0, p.name),
        reverse=True,
    )
    latest = pick_latest_by_filename(files_sorted)

    def on_change_selected_file():
        st.session_state["selected_prof_csv"] = st.session_state["home_file_select"]

    selected_file = st.selectbox(
        "読み込みCSV（デフォルトは最新）",
        options=files_sorted,
        index=files_sorted.index(latest),
        format_func=lambda p: p.name,
        key="home_file_select",
        on_change=on_change_selected_file,
    )

    # 初回起動時の None ガード
    if st.session_state["selected_prof_csv"] is None:
        st.session_state["selected_prof_csv"] = latest

    selected_file = st.session_state["selected_prof_csv"]

    if st.button("🔄 再読み込み（キャッシュクリア）"):
        st.cache_data.clear()

# 最新ファイルの日付（kako_dataの一致に使う）
kaisai_date = extract_yyyymmdd_from_name(selected_file.name)
st.caption(f"参照ファイル: {selected_file.name} （開催日: {kaisai_date if kaisai_date else '-'}）")

# prof_result 側: 場所×Rごとに「総合利益度>=17がいるか」を判定
df_work = df.copy()
if "総合利益度" in df_work.columns:
    df_work["総合利益度"] = pd.to_numeric(df_work["総合利益度"], errors="coerce")
high17_map = (
    df_work.groupby(["場所", "R"])["総合利益度"]
    .apply(lambda s: bool((s >= 17).any()))
    .to_dict()
) if "総合利益度" in df_work.columns else {}

st.info("強調ルール：🔥 Lv4/Lv5  |  ⭐ Lv3 かつ 総合利益度>=17の馬がいるレース")
st.caption(
    "凡例（買い条件バッジ）: ✅=単勝&複勝が同一馬で条件合致 / 🅰️=単勝のみ / 🅱️=複勝のみ  ※[推:〇〇]=推定人気ベースの判定"
)
st.caption(
    "※ このバッジ判定は index_view と同じロジックで、合格馬（総合利益度>=0）を対象にしています。"
)

# -----------------------------------
# 開催レース一覧（netkeiba風）
# -----------------------------------
place_groups = {
    place: sorted(g["R"].dropna().astype(int).unique())
    for place, g in df.groupby("場所")
}

places = list(place_groups.keys())

# 開催場の数だけ横カラムを作る
cols = st.columns(len(places), gap="large")

st.page_link(
    "pages/analysis.py",
    label="指数分析ページ",
    icon="📈",
    use_container_width=True,
    query_params={"csv": selected_file.name}
)

for col, place in zip(cols, places):
    with col:
        st.markdown(f"## 📍 {place}")

        # 1～12Rを「ボタン風」に整列（3列×複数行）
        race_list = place_groups[place]
        n_cols = 1

        for i in range(0, len(race_list), n_cols):
            row = race_list[i:i+n_cols]
            grid_cols = st.columns(n_cols, gap="small")

            for c, r in zip(grid_cols, row):
                with c:
                    lv = level_map.get((place, int(r)))
                    has17 = high17_map.get((place, int(r)), False)

                    # 強調判定
                    highlight_fire = (lv in ("Lv4", "Lv5"))
                    highlight_star = (lv == "Lv3" and has17)

                    if highlight_fire:
                        icon = "🔥"
                    elif highlight_star:
                        icon = "⭐"
                    else:
                        icon = "📊"
                    
                    lv_text = f" {lv}" if lv else ""
                    badge_pair = race_has_condition.get((place, int(r)), ("", ""))
                    badge, badge_est = badge_pair if isinstance(badge_pair, tuple) else (badge_pair, "")
                    est_suffix = f" [推:{badge_est.strip()}]" if badge_est.strip() else ""
                    label = f"{icon} {int(r)}R{lv_text}{badge}{est_suffix}"

                    st.page_link(
                        "pages/index_view.py",
                        label=label,
                        icon=None,
                        query_params={
                            "place": place,
                            "race": int(r),
                            "date": kaisai_date,
                        },
                        use_container_width=True,
                        help=f"Lv={lv if lv else '-'} / 総合利益度>=17: {'あり' if has17 else 'なし'}",
                    )

        st.divider()