import numpy as np
import pandas as pd
from pathlib import Path

STYLE_LABELS = ["逃げ", "先行", "差し", "追込"]

# (距離の上限, 着回数ブロックの距離帯サフィックス)
DISTANCE_BRACKETS = [
    (1200, "1200以下"), (1400, "1201-1400"), (1600, "1401-1600"),
    (1800, "1601-1800"), (2000, "1801-2000"), (2200, "2001-2200"),
    (2400, "2201-2400"), (2800, "2401-2800"), (float("inf"), "2801以上"),
]

SURFACE_SUFFIX = {"芝": "芝", "ダート": "ダ", "障害": "障"}

CK_KEY_FIELDS = ["開催年", "開催月日", "競馬場コード", "開催回", "開催日目", "レース番号", "血統登録番号"]


def load_ck_data(path: Path) -> pd.DataFrame:
    """妙味度指数_jvlink/data_out/ck.csv を読み込む（キー列は文字列のまま保持）"""
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str)


def _distance_bracket(distance) -> str | None:
    d = pd.to_numeric(distance, errors="coerce")
    if pd.isna(d):
        return None
    for upper, label in DISTANCE_BRACKETS:
        if d <= upper:
            return label
    return None


def _place_rate(row: pd.Series, prefix: str) -> float:
    """{prefix}_1〜{prefix}_6（1着〜5着+着外）から複勝率(1-3着/合計)を計算する"""
    counts = [pd.to_numeric(row.get(f"{prefix}_{i}"), errors="coerce") for i in range(1, 7)]
    counts = [c if pd.notna(c) else 0 for c in counts]
    total = sum(counts)
    if total <= 0:
        return np.nan
    return sum(counts[:3]) / total * 100.0


def compute_running_style(row: pd.Series) -> object:
    """脚質傾向_1〜4（逃げ,先行,差し,追込の回数）から最頻の脚質を返す"""
    counts = [pd.to_numeric(row.get(f"脚質傾向_{i}"), errors="coerce") for i in range(1, 5)]
    counts = [c if pd.notna(c) else 0 for c in counts]
    if sum(counts) == 0:
        return np.nan
    return STYLE_LABELS[int(np.argmax(counts))]


def compute_course_aptitude(row: pd.Series) -> float:
    """出走する場所・種別に対応する着回数ブロックから複勝率を計算する（例: 東京芝・着回数）"""
    surface = SURFACE_SUFFIX.get(row.get("種別"))
    place = row.get("場所")
    if surface is None or pd.isna(place):
        return np.nan
    return _place_rate(row, f"{place}{surface}・着回数")


def compute_distance_aptitude(row: pd.Series) -> float:
    """出走する種別・距離に対応する着回数ブロックから複勝率を計算する（障害は距離帯区分が無いため対象外）"""
    surface = {"芝": "芝", "ダート": "ダ"}.get(row.get("種別"))
    if surface is None:
        return np.nan
    bracket = _distance_bracket(row.get("距離"))
    if bracket is None:
        return np.nan
    return _place_rate(row, f"{surface}{bracket}・着回数")


def attach_ck_features(entries: pd.DataFrame, ck_df: pd.DataFrame) -> pd.DataFrame:
    """
    entries: 開催年,開催月日,競馬場コード,開催回,開催日目,レース番号,血統登録番号,場所,種別,距離 を含む当日出走馬DataFrame
    ck_df: load_ck_data() の戻り値
    戻り値: entries に 脚質傾向・コース複勝率・距離帯複勝率 を追加したDataFrame
    """
    ck_small = ck_df.drop_duplicates(subset=CK_KEY_FIELDS)
    merged = entries.merge(ck_small, on=CK_KEY_FIELDS, how="left", suffixes=("", "_ck"))

    merged["脚質傾向"] = merged.apply(compute_running_style, axis=1)
    merged["コース複勝率"] = merged.apply(compute_course_aptitude, axis=1)
    merged["距離帯複勝率"] = merged.apply(compute_distance_aptitude, axis=1)

    keep = [c for c in entries.columns] + ["脚質傾向", "コース複勝率", "距離帯複勝率"]
    return merged[keep]
