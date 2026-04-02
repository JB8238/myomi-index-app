import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import re
from datetime import datetime


def extract_yyyymmdd_from_name_fiv(filename: str) -> int | None:
    candidates = re.findall(r"\d{8}", filename)
    valid = []
    for s in candidates:
        try:
            datetime.strptime(s, "%Y%m%d")
            valid.append(int(s))
        except ValueError:
            pass
    return max(valid) if valid else None

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

# --------------------------------------
# prof_result 全CSVから「前走総合利益度」を引くための履歴テーブル
# --------------------------------------
@st.cache_data(show_spinner="📚 prof_result 履歴を整備しています…")
def build_prof_history(data_dir_str: str) -> pd.DataFrame:
    data_dir = Path(data_dir_str)
    files = sorted([p for p in data_dir.rglob("*.csv") if p.is_file()])
    rows = []
    for p in files:
        date_int = extract_yyyymmdd_from_name_fiv(p.name)
        if date_int is None:
            continue
        mtime = p.stat().st_mtime    # 「日時」に近い情報として更新時刻を併用
        try:
            df0 = pd.read_csv(str(p), encoding="cp932")
        except Exception:
            # 文字コード差分がある場合に備えて最低限
            df0 = pd.read_csv(str(p), encoding="cp932", errors="ignore")
        
        if "馬名" not in df0.columns or "総合利益度" not in df0.columns:
            continue

        # 必要最低限の列に絞る（無い列は落ちないように）
        keep = [c for c in ["場所", "R", "馬番", "馬名", "総合利益度"] if c in df0.columns]
        df1 = df0[keep].copy()
        df1["__date"] = int(date_int)
        df1["__mtime"] = float(mtime)
        df1["__file"] = p.name

        # 型を整える
        if "R" in df1.columns:
            df1["R"] = pd.to_numeric(df1["R"], errors="coerce")
        if "馬番" in df1.columns:
            df1["馬番"] = pd.to_numeric(df1["馬番"], errors="coerce")
        df1["総合利益度"] = pd.to_numeric(df1["総合利益度"], errors="coerce")
        df1["馬名"] = df1["馬名"].astype(str)

        rows.append(df1)
    
    if not rows:
        return pd.DataFrame(columns=["馬名", "総合利益度", "__date", "__mtime", "__file"])
    
    hist = pd.concat(rows, ignore_index=True)
    # 馬名が空の行は除外
    hist = hist[hist["馬名"].notna() & (hist["馬名"].astype(str).str.strip() != "")]
    return hist

def find_prev_total(history: pd.DataFrame, horse_name: str, cur_date: int, cur_mtime: float, cur_file: str):
    h = history[
        (history["馬名"] == horse_name) &
        (history["__file"] != cur_file)
    ]
    if h.empty:
        return np.nan
    h = h[
        (h["__date"] < cur_date) |
        ((h["__date"] == cur_date) & (h["__mtime"] < cur_mtime))
    ]
    if h.empty:
        return np.nan
    h = h.sort_values(["__date", "__mtime"])
    return h.iloc[-1]["総合利益度"]