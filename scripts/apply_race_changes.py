"""
scripts/apply_race_changes.py

fetch_netkeiba_changes.py が出力した変更 dict をもとに
preprocessed_data_YYYYMMDD.csv を更新するユーティリティ。

Streamlit ページから直接インポートして使用する。
バックアップファイルを同ディレクトリに作成してから上書きする。
"""

import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_condition(val: str) -> str:
    """馬場状態から (暫定)/(確定) を除去して正規化する。"""
    if not isinstance(val, str):
        return ""
    return val.replace("(暫定)", "").replace("(確定)", "").strip()


def is_bad_track(condition: str) -> bool:
    """道悪かどうかを判定する（preprocessing.py の .*良.* ロジックと同一）。"""
    return not bool(re.search(r".*良.*", condition))


def apply_changes(kaisai_date: str, changes: dict) -> dict:
    """
    変更 dict を preprocessed_data_YYYYMMDD.csv に適用する。

    馬場状態の変更:
        - 馬場状態列を新しい条件に更新
        - 道悪判定列を再計算（良→NaN、それ以外→「芝道悪」or「ダ道悪」）
    騎手変更:
        - 騎手名列を新しい騎手名に更新

    Returns:
        {
            "success": bool,
            "applied_track": int,   # 馬場状態変更で更新した行数
            "applied_jockey": int,  # 騎手変更で更新した行数
            "backup_path": str,     # バックアップファイルのパス
            "error": str | None,
        }
    """
    year = kaisai_date[:4]
    csv_path = Path(f"data/{year}/{kaisai_date}/preprocessed_data_{kaisai_date}.csv")

    if not csv_path.exists():
        return {
            "success": False,
            "applied_track": 0,
            "applied_jockey": 0,
            "backup_path": "",
            "error": f"CSVファイルが見つかりません: {csv_path}",
        }

    # 更新前にバックアップを作成
    backup_path = csv_path.with_suffix(
        f".bak_{datetime.now().strftime('%H%M%S')}.csv"
    )
    shutil.copy2(csv_path, backup_path)

    df = pd.read_csv(csv_path, encoding="utf-8")

    # 道悪判定・騎手名列を object 型にしておく（NaN 列への文字列代入で dtype 不整合を防ぐ）
    df["道悪判定"] = df["道悪判定"].astype(object)
    df["騎手名"] = df["騎手名"].astype(object)

    applied_track = 0
    applied_jockey = 0

    # --- 馬場状態の更新 ---
    for change in changes.get("track_changes", []):
        venue = change["場所"]
        surface_type = change["種別"]
        new_cond = change["新馬場状態"]

        mask = (df["場所"] == venue) & (df["種別"] == surface_type)
        count = int(mask.sum())
        if count == 0:
            continue

        # 馬場状態を更新
        df.loc[mask, "馬場状態"] = new_cond

        # 道悪判定を再計算（preprocessing.py の .*良.* ロジックと同一）
        if is_bad_track(new_cond):
            # 稍重/重/不良 → 「芝道悪」or「ダ道悪」
            if surface_type == "芝":
                df.loc[mask, "道悪判定"] = "芝道悪"
            else:
                # ダート・障害はどちらも「ダ道悪」（preprocessing.py に準拠）
                df.loc[mask, "道悪判定"] = "ダ道悪"
        else:
            # 良 → 道悪判定をクリア
            df.loc[mask, "道悪判定"] = np.nan

        applied_track += count

    # --- 騎手名の更新 ---
    for change in changes.get("jockey_changes", []):
        venue = change["場所"]
        race_r = change["R"]
        umaban = change["馬番"]
        new_jockey = change["新騎手名"]

        mask = (
            (df["場所"] == venue)
            & (df["R"] == race_r)
            & (df["馬番"] == umaban)
        )
        count = int(mask.sum())
        if count == 0:
            continue

        df.loc[mask, "騎手名"] = new_jockey
        applied_jockey += count

    df.to_csv(csv_path, index=False, encoding="utf-8")

    return {
        "success": True,
        "applied_track": applied_track,
        "applied_jockey": applied_jockey,
        "backup_path": str(backup_path),
        "error": None,
    }
