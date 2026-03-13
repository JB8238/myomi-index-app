import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

DATA_DIR = Path(".", "prof_result")
KAKO_DIR = Path(".", "kako_data")

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

def pick_latest(files):
    dated = []
    for f in files:
        d = extract_yyyymmdd_from_name(f.name)
        if d:
            dated.append((d, f))
    dated.sort()
    return dated[-1][1] if dated else None

def pick_latest_kako_csv(kako_dir: Path, target_date: str | None = None) -> Path | None:
    """kako_data_YYYYMMDD.csv を探し、target_date一致を優先、なければ日付最大"""
    files = sorted([p for p in kako_dir.rglob("kako_data_*.csv") if p.is_file()])
    if not files:
        return None
    dated = []
    for f in files:
        d = extract_yyyymmdd_from_name(f.name)
        if d:
            dated.append((d, f))
    if not dated:
        return None
    if target_date is not None:
        same = [t for t in dated if t[0] == target_date]
        if same:
            same.sort(key=lambda x: x[1].name)
            return same[-1][1]
    dated.sort(key=lambda x: (x[0], x[1].name))
    return dated[-1][1]

@st.cache_data(show_spinner="📥 kako_data を読み込んでいます…")
def load_kako_csv(path_str: str) -> pd.DataFrame:
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path_str, header=None, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path_str, header=None, encoding="cp932", errors="ignore")

# -----------------------------------
# データ読み込み
# -----------------------------------
files = list_csv_files()
latest = pick_latest(files)

if latest is None:
    st.error("prof_result にCSVが見つかりません")
    st.stop()

df = pd.read_csv(latest, encoding="cp932")

if not {"場所", "R"}.issubset(df.columns):
    st.error("CSVに「場所」「R」列が必要です")
    st.stop()

# 最新ファイルの日付（kako_dataの一致に使う）
kaisai_date = extract_yyyymmdd_from_name(latest.name)
st.caption(f"参照ファイル: {latest.name} （開催日: {kaisai_date if kaisai_date else '-'}）")

# prof_result 側: 場所×Rごとに「総合利益度>=17がいるか」を判定
df_work = df.copy()
if "総合利益度" in df_work.columns:
    df_work["総合利益度"] = pd.to_numeric(df_work["総合利益度"], errors="coerce")
high17_map = (
    df_work.groupby(["場所", "R"])["総合利益度"]
    .apply(lambda s: bool((s >= 17).any()))
    .to_dict()
) if "総合利益度" in df_work.columns else {}

# kako_data 側: 場所(D=3)×R(E=4)ごとにレースレベル(L=11)を取得
level_map = {}
if KAKO_DIR.exists():
    kako_path = pick_latest_kako_csv(KAKO_DIR, target_date=kaisai_date)
    if kako_path is not None:
        df_kako = load_kako_csv(str(kako_path))
        # D列=3, E列=4, L列=11（ヘッダなし）
        tmp = df_kako[[3, 4, 11]].copy() if df_kako.shape[1] >= 12 else None
        if tmp is not None:
            tmp.columns = ["場所", "R", "Lv"]
            tmp["R"] = pd.to_numeric(tmp["R"], errors="coerce")
            tmp["Lv"] = tmp["Lv"].astype(str).str.strip()
            # 場所×Rごとの代表Lv（複数行ある場合は最頻）
            for (p, r), g in tmp.dropna(subset=["R"]).groupby(["場所", "R"]):
                lv_series = g["Lv"].dropna()
                if not lv_series.empty:
                    level_map[(p, int(r))] = lv_series.mode().iloc[0]

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
                        query_params={"place": place, "race": int(r)},
                        use_container_width=True,
                        help=f"Lv={lv if lv else '-'} / 総合利益度>=17: {'あり' if has17 else 'なし'}",
                    )

        st.divider()