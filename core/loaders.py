import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import re
from datetime import datetime


def list_preprocessed_files(data_dir: Path):
    return sorted([p for p in data_dir.rglob("preprocessed_data_*.csv") if p.is_file()])

def extract_yyyymmdd_from_name(filename: str) -> int | None:
    m = re.findall(r"(\d{8})", filename)
    for s in reversed(m):
        try:
            datetime.strptime(s, "%Y%m%d")
            return int(s)
        except ValueError:
            pass
    return None


@st.cache_data(show_spinner="📥 preprocessed_data 読み込み中…")
def load_preprocessed(data_dir: Path) -> pd.DataFrame:
    rows = []
    for p in list_preprocessed_files(data_dir):
        d = extract_yyyymmdd_from_name(p.name)
        if d is None:
            continue

        df0 = pd.read_csv(p, encoding="utf-8")
        df0.columns = [str(c).strip() for c in df0.columns]

        if not {"場所", "R", "馬番", "レースレベル"}.issubset(df0.columns):
            continue

        tmp = df0[["場所", "R", "馬番", "レースレベル"]].copy()
        tmp["開催日"] = d

        # 正規化
        tmp["場所"] = tmp["場所"].astype(str).str.replace("\u3000", " ").str.strip()
        tmp["レースレベル"] = tmp["レースレベル"].astype(str).str.strip()

        tmp["R"] = pd.to_numeric(tmp["R"], errors="coerce")
        tmp["馬番"] = pd.to_numeric(tmp["馬番"], errors="coerce")

        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["開催日", "場所", "R", "馬番", "レースレベル"])

    return pd.concat(rows, ignore_index=True)

@st.cache_data(show_spinner="📥 return_data 読み込み中…")
def load_return(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    df.rename(columns={"Ｒ": "R"}, inplace=True)

    RESULT_COLS = [
        "人気",
        "単オッズ",
        "複勝オッズ下",
        "上",
        "単勝",
        "複勝",
        "馬連配当",
    ]

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

@st.cache_data(show_spinner="📥 preprocessed_data 読み込み中…")
def load_preprocessed_for_race(prep_dir: Path, target_date: int) -> pd.DataFrame:
    files = sorted(prep_dir.rglob("preprocessed_data_*.csv"))
    rows = []

    for p in files:
        d = extract_yyyymmdd_from_name(p.name)
        if d != target_date:
            continue

        df0 = pd.read_csv(p, encoding="utf-8")
        df0.columns = [str(c).strip() for c in df0.columns]

        if not {"場所", "R", "レースレベル"}.issubset(df0.columns):
            continue

        tmp = df0[["場所", "R", "レースレベル"]].copy()
        tmp["開催日"] = d

        tmp["場所"] = (
            tmp["場所"].astype(str)
            .str.replace("\u3000", " ")
            .str.strip()
        )
        tmp["R"] = pd.to_numeric(tmp["R"], errors="coerce")
        tmp["レースレベル"] = tmp["レースレベル"].astype(str).str.strip()

        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["開催日", "場所", "R", "レースレベル"])

    return pd.concat(rows, ignore_index=True)

@st.cache_data(show_spinner="📥 CSVを読み込んでいます…")
def load_csv(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str, encoding="cp932")
    return df

# ---------------------------
# 型整形・欠損整備
# ---------------------------
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    num_cols = [
        "R", "馬番",
        "騎手利益度", "騎手利益度順位",
        "種牡馬利益度", "種牡馬利益度順位",
        "調教師利益度", "調教師利益度順位",
        "総合利益度", "総合利益度順位",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df