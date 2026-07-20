"""券種ごとの統一ベットテーブル構築。

すべての券種を「1件=1ベット」の共通スキーマに正規化する:
    開催日, 場所, R, レースレベル, cost(円), return(円, ハズレ0), <候補特徴量...>

候補特徴量は生の数値のまま持たせる（ビン留めは core.strategy_engine 側で
学習データにfitしたqcut境界を使って行う。ここでは切らない）。

※ 単オッズ・複勝オッズ下・確定人気は return_data（レース結果ファイル）由来で、
  レース終了後にしか確定しない。当日の推奨に使えないため、候補特徴量からは
  意図的に外している。「人気乖離」は常に推定人気（レース前に分かる）ベースで計算した
  値をこの列名で渡す前提（呼び出し側の責務）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CANDIDATE_FEATURE_COLS = [
    "利益度上昇値", "人気乖離", "cv", "合格数区分", "偏差値合格数区分",
]


# =========================================================
# 馬連 的中ペア抽出
# =========================================================
def extract_hit_pairs(df_return: pd.DataFrame) -> pd.DataFrame:
    """
    return_data_merged 由来のデータから、レースごとの馬連的中ペアと配当を抽出する。
    馬連配当は的中した2頭の行それぞれに同じ値が入る形式のため、
    (開催日,場所,R) でグルーピングしてちょうど2件になるレースのみを対象にする
    （デッドヒート等で3件以上になる稀なケースはペアを一意に特定できないため除外）。
    """
    key = ["開催日", "場所", "R"]
    need = set(key) | {"馬番", "馬連配当"}
    if not need.issubset(df_return.columns):
        return pd.DataFrame(columns=key + ["horse_a", "horse_b", "payout"])

    d = df_return.dropna(subset=["馬連配当"]).copy()
    d["馬連配当"] = pd.to_numeric(d["馬連配当"], errors="coerce")
    d["馬番"] = pd.to_numeric(d["馬番"], errors="coerce")
    d = d.dropna(subset=["馬連配当", "馬番"])

    rows = []
    for (d_, p_, r_), g in d.groupby(key, observed=True):
        if len(g) != 2:
            continue
        horses = sorted(g["馬番"].astype(int).tolist())
        rows.append({
            "開催日": d_, "場所": p_, "R": r_,
            "horse_a": horses[0], "horse_b": horses[1],
            "payout": float(g["馬連配当"].iloc[0]),
        })

    return pd.DataFrame(rows, columns=key + ["horse_a", "horse_b", "payout"])


# =========================================================
# 単勝・複勝
# =========================================================
def build_win_bet_table(df_pop: pd.DataFrame) -> pd.DataFrame:
    """df_pop: 合格馬(総合利益度>=0)にあらかじめ絞り込み済みの馬単位テーブルを想定"""
    need = {"開催日", "場所", "R", "レースレベル", "単勝"}
    if not need.issubset(df_pop.columns):
        return pd.DataFrame()

    d = df_pop
    out = pd.DataFrame({
        "開催日": d["開催日"],
        "場所": d["場所"],
        "R": d["R"],
        "レースレベル": d["レースレベル"],
        "cost": 100.0,
        "return": pd.to_numeric(d["単勝"], errors="coerce").fillna(0.0),
    })
    for c in CANDIDATE_FEATURE_COLS:
        out[c] = d[c] if c in d.columns else np.nan
    return out


def build_place_bet_table(df_pop: pd.DataFrame) -> pd.DataFrame:
    """df_pop: 合格馬(総合利益度>=0)にあらかじめ絞り込み済みの馬単位テーブルを想定"""
    need = {"開催日", "場所", "R", "レースレベル", "複勝"}
    if not need.issubset(df_pop.columns):
        return pd.DataFrame()

    d = df_pop
    out = pd.DataFrame({
        "開催日": d["開催日"],
        "場所": d["場所"],
        "R": d["R"],
        "レースレベル": d["レースレベル"],
        "cost": 100.0,
        "return": pd.to_numeric(d["複勝"], errors="coerce").fillna(0.0),
    })
    for c in CANDIDATE_FEATURE_COLS:
        out[c] = d[c] if c in d.columns else np.nan
    return out


# =========================================================
# 馬連（軸流し）: 軸=総合利益度順位1位・合格馬 固定、相手候補が1ベット
# =========================================================
def build_nagashi_bet_table(df_pop: pd.DataFrame, hit_pairs: pd.DataFrame) -> pd.DataFrame:
    key = ["開催日", "場所", "R"]
    need = set(key) | {"馬番", "総合利益度", "総合利益度順位", "レースレベル"}
    if not need.issubset(df_pop.columns):
        return pd.DataFrame()

    d = df_pop.copy()
    d["総合利益度順位"] = pd.to_numeric(d["総合利益度順位"], errors="coerce")
    d["総合利益度"] = pd.to_numeric(d["総合利益度"], errors="coerce")
    d["馬番"] = pd.to_numeric(d["馬番"], errors="coerce")

    axis = d[(d["総合利益度順位"] == 1) & (d["総合利益度"] >= 0)]
    axis = axis.drop_duplicates(subset=key, keep="first")[key + ["馬番"]].rename(columns={"馬番": "軸馬番"})
    if axis.empty:
        return pd.DataFrame()

    bet = d.merge(axis, on=key, how="inner")
    bet = bet[bet["馬番"] != bet["軸馬番"]].copy()
    bet = bet.rename(columns={"馬番": "相手馬番"})

    if not hit_pairs.empty:
        bet = bet.merge(hit_pairs, on=key, how="left")
        axis_i = bet["軸馬番"].astype("Int64")
        partner_i = bet["相手馬番"].astype("Int64")
        is_hit = ((axis_i == bet["horse_a"]) & (partner_i == bet["horse_b"])) | \
                 ((axis_i == bet["horse_b"]) & (partner_i == bet["horse_a"]))
        payout = pd.to_numeric(bet["payout"], errors="coerce").fillna(0.0)
        bet["return"] = np.where(is_hit.fillna(False), payout, 0.0)
    else:
        bet["return"] = 0.0
    bet["cost"] = 100.0

    keep = key + ["レースレベル", "軸馬番", "相手馬番", "cost", "return"] + CANDIDATE_FEATURE_COLS
    keep = [c for c in keep if c in bet.columns]
    return bet[keep]


# =========================================================
# 馬連（ボックス）: 総合利益度順位の上位N頭ボックス
# =========================================================
def build_box_bet_table(df_pop: pd.DataFrame, hit_pairs: pd.DataFrame, box_sizes=(2, 3, 4, 5)) -> pd.DataFrame:
    """
    総合利益度順位のみで上位を決める（合格馬フィルタは呼び出し側の母集団に委ねる＝相対順位の戦略）。
    """
    key = ["開催日", "場所", "R"]
    need = set(key) | {"馬番", "総合利益度順位", "レースレベル"}
    if not need.issubset(df_pop.columns):
        return pd.DataFrame()

    d = df_pop.copy()
    d["総合利益度順位"] = pd.to_numeric(d["総合利益度順位"], errors="coerce")
    d["馬番"] = pd.to_numeric(d["馬番"], errors="coerce")

    hp_map = {}
    if not hit_pairs.empty:
        for _, r in hit_pairs.iterrows():
            hp_map[(r["開催日"], r["場所"], r["R"])] = (int(r["horse_a"]), int(r["horse_b"]), float(r["payout"]))

    rows = []
    for (d_, p_, r_), g in d.groupby(key, observed=True):
        g2 = g.dropna(subset=["総合利益度順位", "馬番"]).sort_values("総合利益度順位")
        if g2.empty:
            continue
        race_level = g2["レースレベル"].iloc[0]
        cv_val = pd.to_numeric(g2["cv"], errors="coerce").iloc[0] if "cv" in g2.columns else np.nan
        hit = hp_map.get((d_, p_, r_))

        for n in box_sizes:
            if len(g2) < n:
                continue
            top = set(g2.iloc[:n]["馬番"].astype(int).tolist())
            cost = 100.0 * (n * (n - 1) // 2)
            is_hit = hit is not None and hit[0] in top and hit[1] in top
            rows.append({
                "開催日": d_, "場所": p_, "R": r_,
                "レースレベル": race_level,
                "box_N": n,
                "cv": cv_val,
                "cost": cost,
                "return": float(hit[2]) if is_hit else 0.0,
            })

    return pd.DataFrame(rows)
