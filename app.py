import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

DATA_DIR = Path("prof_result")
KAKO_DIR = Path("kako_data")
PREP_DIR = Path("data")

st.set_page_config(
    page_title="開催レース一覧",
    page_icon="🏇",
    layout="wide",
)

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

if selected_file is None:
    st.error("prof_result にCSVが見つかりません")
    st.stop()


# レースレベル（preprocessed_data 由来）
level_map = load_race_level_map(PREP_DIR, kaisai_date)

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

    selected_file = st.selectbox(
        "読み込みCSV（デフォルトは最新）",
        options=files_sorted,
        index=files_sorted.index(latest),
        format_func=lambda p: p.name,
        key="home_file_select",
    )

    st.session_state["selected_prof_csv"] = selected_file

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
                    label = f"{icon} {int(r)}R{lv_text}"

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