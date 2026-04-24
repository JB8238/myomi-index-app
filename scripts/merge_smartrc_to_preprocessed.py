"""
scripts/merge_smartrc_to_preprocessed.py

smartrc_YYYYMMDD.csv の「推定人気」「人気ランク」を
preprocessed_data_YYYYMMDD.csv に結合して上書き保存する。

Usage:
    python scripts/merge_smartrc_to_preprocessed.py

対応モード:
    1: 単日
    2: 期間指定（data/smartrc/ にある全CSVを対象に自動スキャン）

結合キー: 場所, R, 馬番
"""

import sys
from pathlib import Path

import pandas as pd

SMARTRC_DIR = Path("data/smartrc")
DATA_DIR    = Path("data")

VENUE_CODE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


def preprocessed_path(kaisai_date: str) -> Path:
    year = kaisai_date[:4]
    return DATA_DIR / year / kaisai_date / f"preprocessed_data_{kaisai_date}.csv"


def merge_one(kaisai_date: str) -> str:
    """
    1開催日分をマージして preprocessed_data を上書き。
    戻り値: 結果メッセージ
    """
    smartrc_path = SMARTRC_DIR / f"smartrc_{kaisai_date}.csv"
    prep_path    = preprocessed_path(kaisai_date)

    if not smartrc_path.exists():
        return f"  スキップ: smartrc_{kaisai_date}.csv が存在しません"
    if not prep_path.exists():
        return f"  スキップ: preprocessed_data_{kaisai_date}.csv が存在しません"

    # --- 読み込み ---
    df_smartrc = pd.read_csv(smartrc_path, encoding="utf-8-sig")
    df_prep    = pd.read_csv(prep_path,    encoding="utf-8")

    # --- 型と表記を揃える ---
    for df in [df_smartrc, df_prep]:
        if "場所" in df.columns:
            df["場所"] = df["場所"].astype(str).str.replace("　", " ").str.strip()
        if "R" in df.columns:
            df["R"] = pd.to_numeric(df["R"], errors="coerce")
        if "馬番" in df.columns:
            df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")

    if "推定人気" in df_smartrc.columns:
        df_smartrc["推定人気"] = pd.to_numeric(df_smartrc["推定人気"], errors="coerce")

    # --- 既存列を一旦削除してから結合（重複防止） ---
    for col in ["推定人気", "人気ランク"]:
        if col in df_prep.columns:
            df_prep = df_prep.drop(columns=[col])

    merge_cols = ["場所", "R", "馬番", "推定人気", "人気ランク"]
    src = df_smartrc[[c for c in merge_cols if c in df_smartrc.columns]].copy()

    df_merged = df_prep.merge(src, on=["場所", "R", "馬番"], how="left")

    # --- 上書き保存（utf-8、BOM なし） ---
    df_merged.to_csv(prep_path, index=False, encoding="utf-8")

    n_matched = df_merged["推定人気"].notna().sum() if "推定人気" in df_merged.columns else 0
    return f"  ✓ {prep_path.name}  ({n_matched} 頭に推定人気を付与)"


def available_smartrc_dates() -> list[str]:
    """data/smartrc/ にある日付リストを返す（昇順）"""
    dates = []
    for p in SMARTRC_DIR.glob("smartrc_*.csv"):
        stem = p.stem  # smartrc_YYYYMMDD
        d = stem.replace("smartrc_", "")
        if len(d) == 8 and d.isdigit():
            dates.append(d)
    return sorted(dates)


def main():
    print("マージモードを選択してください")
    print("  1: 単日")
    print("  2: 期間指定")
    mode = input("モード (1 or 2): ").strip()

    if mode == "1":
        date_input = input("開催日 (例: 20260419): ").strip()
        if len(date_input) != 8 or not date_input.isdigit():
            print("日付の形式が正しくありません。")
            sys.exit(1)
        target_dates = [date_input]

    elif mode == "2":
        start_input = input("開始日 (例: 20260101): ").strip()
        end_input   = input("終了日 (例: 20260419): ").strip()
        if (len(start_input) != 8 or not start_input.isdigit() or
                len(end_input) != 8 or not end_input.isdigit()):
            print("日付の形式が正しくありません。")
            sys.exit(1)
        all_dates = available_smartrc_dates()
        target_dates = [d for d in all_dates if start_input <= d <= end_input]
        if not target_dates:
            print("指定期間内に smartrc CSV が見つかりませんでした。")
            sys.exit(1)
        print(f"{len(target_dates)} 日分が対象: {', '.join(target_dates)}")
    else:
        print("1 か 2 を入力してください。")
        sys.exit(1)

    success, skipped = 0, 0
    for d in target_dates:
        print(f"\n[{d}]")
        msg = merge_one(d)
        print(msg)
        if "✓" in msg:
            success += 1
        else:
            skipped += 1

    print(f"\n{'='*40}")
    print(f"完了: 成功 {success} 日 / スキップ {skipped} 日")


if __name__ == "__main__":
    main()
