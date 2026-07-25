import numpy as np
import pandas as pd
from pathlib import Path


def load_horse_race_history(path: Path) -> pd.DataFrame:
    """妙味度指数_jvlink/data_out/horse_race_history.csv を読み込み、日付列を付与する"""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df["開催月日"] = df["開催月日"].astype(str).str.zfill(4)
    df["date"] = pd.to_datetime(df["開催年"] + df["開催月日"], format="%Y%m%d", errors="coerce")
    df["距離"] = pd.to_numeric(df["距離"], errors="coerce")
    return df


def build_zensou_features(today_df: pd.DataFrame, history_df: pd.DataFrame, kaisai_date: str) -> pd.DataFrame:
    """
    馬別レース履歴（JV-Link由来）から、preprocessing.pyの旧zensou_df_pre相当の
    DataFrame（距離,馬名,前走間隔,前走距離,前-2走前間隔,2-3走前間隔）を組み立てる。

    JV-DataのSEレコード自体には前走情報が無いため、血統登録番号ごとに当日より前の
    レース日を時系列で遡って算出する。「前走間隔」等はTARGETのkako_data列と同じ
    「レース日程の週数差」（カレンダー週数、"連闘"=1週差 相当）で揃えており、後段の
    臨戦過程・距離変遷判定ロジック（-1補正込み）にそのまま渡せる。

    today_df: 血統登録番号,馬名,距離（今回のレース距離）を持つ当日出走馬DataFrame
    """
    cur_date = pd.to_datetime(kaisai_date, format="%Y%m%d")
    hist_before = history_df[history_df["date"] < cur_date]
    by_horse = {
        ped_id: g.sort_values("date", ascending=False)
        for ped_id, g in hist_before.groupby("血統登録番号")
    }

    def weeks_gap(a, b):
        return round((a - b).days / 7)

    rows = []
    for _, r in today_df.iterrows():
        h = by_horse.get(r["血統登録番号"])

        interval1 = interval2 = interval3 = np.nan
        dist1 = np.nan
        if h is not None and len(h) > 0:
            dates = h["date"].tolist()
            dists = h["距離"].tolist()
            interval1 = weeks_gap(cur_date, dates[0])
            dist1 = dists[0]
            if len(h) >= 2:
                interval2 = weeks_gap(dates[0], dates[1])
            if len(h) >= 3:
                interval3 = weeks_gap(dates[1], dates[2])

        rows.append({
            "距離": r["距離"],
            "馬名": r["馬名"],
            "前走間隔": interval1,
            "前走距離": dist1,
            "前-2走前間隔": interval2,
            "2-3走前間隔": interval3,
        })

    return pd.DataFrame(rows)
