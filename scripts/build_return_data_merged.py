import pandas as pd
from pathlib import Path
import re
from datetime import datetime

SRC_ROOT = Path(r"C:\TFJV\TXT\data\return_data")
OUT_PATH = Path("./data/return_data_merged.csv")

USE_COLS = [
    "場所",
    "R",
    "馬番",
    "人気",
    "単オッズ",
    "複勝オッズ下",
    "上",
    "単勝",
    "複勝",
    "馬連配当",
]

# smartrc からマージする列（キー列を除く）
SMARTRC_MERGE_COLS = [
    "推定人気", "人気ランク",
    "前走_評価", "前々走_評価", "3走前_評価", "4走前_評価", "5走前_評価",
]

def extract_yyyymmdd(name: str):
    m = re.findall(r"(\d{8})", name)
    for s in reversed(m):
        try:
            datetime.strptime(s, "%Y%m%d")
            return int(s)
        except ValueError:
            pass
    return None

if not SRC_ROOT.exists():
    raise FileNotFoundError(f"SRC_ROOT does not exist: {SRC_ROOT}")

rows = []
scanned = 0
accepted = 0

for year_dir in SRC_ROOT.glob("*"):
    if not year_dir.is_dir():
        continue

    for csv_path in year_dir.glob("return_data_*.csv"):
        scanned += 1
        d = extract_yyyymmdd(csv_path.name)
        if d is None:
            print(f"⚠ 日付取得不可: {csv_path.name}")
            continue

        df = pd.read_csv(csv_path, encoding="cp932")
        df.columns = [str(c).strip() for c in df.columns]

        # ★ 先に列名正規化（最重要）
        df.rename(columns={"Ｒ": "R"}, inplace=True)

        # 必須列チェック（正規化後）
        required = {"場所", "R", "馬番"}
        if not required.issubset(df.columns):
            print(f"⚠ 必須列不足でスキップ: {csv_path.name}")
            print(f"   columns={list(df.columns)}")
            continue

        keep = [c for c in USE_COLS if c in df.columns]
        df = df[keep].copy()

        df["開催日"] = d
        df["R"] = pd.to_numeric(df["R"], errors="coerce")
        df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")

        rows.append(df)
        accepted += 1

print(f"scanned files : {scanned}")
print(f"accepted files: {accepted}")

if not rows:
    raise RuntimeError(
        "No valid return_data CSVs were found.\n"
        "Check directory structure, filenames, and required columns."
    )

merged = pd.concat(rows, ignore_index=True)

# ---- smartrc データのマージ ----
SMARTRC_DIR = Path("./data/smartrc")
KEY_COLS = ["開催日", "場所", "R", "馬番"]

smartrc_rows = []
for csv_path in sorted(SMARTRC_DIR.glob("smartrc_*.csv")):
    d = extract_yyyymmdd(csv_path.name)
    if d is None:
        continue
    df_s = pd.read_csv(csv_path, encoding="utf-8-sig")
    df_s["開催日"] = d
    df_s["R"] = pd.to_numeric(df_s["R"], errors="coerce")
    df_s["馬番"] = pd.to_numeric(df_s["馬番"], errors="coerce")
    if "推定人気" in df_s.columns:
        df_s["推定人気"] = pd.to_numeric(df_s["推定人気"], errors="coerce")

    src_cols = KEY_COLS + [c for c in SMARTRC_MERGE_COLS if c in df_s.columns]
    smartrc_rows.append(df_s[src_cols])

if smartrc_rows:
    df_smartrc_all = pd.concat(smartrc_rows, ignore_index=True)
    df_smartrc_all["場所"] = df_smartrc_all["場所"].astype(str).str.replace("　", " ").str.strip()
    merged["場所"] = merged["場所"].astype(str).str.replace("　", " ").str.strip()

    # 既存列を上書きしないよう drop してから merge
    drop_cols = [c for c in SMARTRC_MERGE_COLS if c in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    merged = merged.merge(
        df_smartrc_all,
        on=KEY_COLS,
        how="left",
    )
    print(f"smartrc files merged: {len(smartrc_rows)} 日分")
else:
    print("smartrc データなし（スキップ）")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"✅ merged return_data written to: {OUT_PATH}")
print(f"rows: {len(merged)}")
