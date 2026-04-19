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

# ---- smartrc 推定人気・人気ランク のマージ ----
SMARTRC_DIR = Path("./data/smartrc")
smartrc_rows = []
for csv_path in sorted(SMARTRC_DIR.glob("smartrc_*.csv")):
    d = extract_yyyymmdd(csv_path.name)
    if d is None:
        continue
    df_s = pd.read_csv(csv_path, encoding="utf-8-sig")
    df_s["開催日"] = d
    df_s["R"] = pd.to_numeric(df_s["R"], errors="coerce")
    df_s["馬番"] = pd.to_numeric(df_s["馬番"], errors="coerce")
    df_s["推定人気"] = pd.to_numeric(df_s["推定人気"], errors="coerce")
    smartrc_rows.append(df_s[["開催日", "場所", "R", "馬番", "推定人気", "人気ランク"]])

if smartrc_rows:
    df_smartrc_all = pd.concat(smartrc_rows, ignore_index=True)
    df_smartrc_all["場所"] = df_smartrc_all["場所"].astype(str).str.replace("\u3000", " ").str.strip()
    merged["場所"] = merged["場所"].astype(str).str.replace("\u3000", " ").str.strip()
    # 既存列を上書きしないよう drop してから merge
    for col in ["推定人気", "人気ランク"]:
        if col in merged.columns:
            merged = merged.drop(columns=[col])
    merged = merged.merge(
        df_smartrc_all,
        on=["開催日", "場所", "R", "馬番"],
        how="left",
    )
    print(f"smartrc files merged: {len(smartrc_rows)} 日分")
else:
    print("smartrc データなし（スキップ）")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"✅ merged return_data written to: {OUT_PATH}")
print(f"rows: {len(merged)}")