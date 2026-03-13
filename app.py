import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

DATA_DIR = Path(".", "prof_result")

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

# -----------------------------------
# 開催レース一覧（netkeiba風）
# -----------------------------------
for place, g in df.groupby("場所"):
    st.subheader(f"📍 {place}")

    races = sorted(g["R"].dropna().astype(int).unique())

    for r in races:
        col1, col2 = st.columns([1, 6])

        with col1:
            st.markdown(f"### {r}R")

        with col2:
            st.page_link(
                "pages/index_viewer.py",
                label="指数を見る",
                icon="📊",
                query_params={
                    "place": place,
                    "race": r,
                },
            )
    
    st.divider()