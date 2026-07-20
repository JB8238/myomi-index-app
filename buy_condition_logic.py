# buy_condition_logic.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from core.strategy_engine import judge


@st.cache_data(show_spinner=False)
def load_buy_conditions(path_str: str, file_mtime: float | None = None, bet_type: str | None = None) -> pd.DataFrame:
    """core.strategy_engine.discover_rules が出力したルールCSVを読み込む（統一ローダ）

    bet_type: 指定すると "bet_type" 列でその券種の行だけに絞り込む（例: "単勝"）
    """
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()

    try:
        dfc = pd.read_csv(p, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    dfc.columns = [str(c).strip() for c in dfc.columns]

    for c in ["件数", "ROI", "CI_low", "CI_high", "p_value",
              "feat1_low", "feat1_high", "feat2_low", "feat2_high"]:
        if c in dfc.columns:
            dfc[c] = pd.to_numeric(dfc[c], errors="coerce")

    for c in ["採用", "多重検定OK", "期間安定OK"]:
        if c in dfc.columns:
            dfc[c] = dfc[c].astype(bool)

    # 自動化スクリプトは「採用」された行のみ書き出す想定だが、念のためここでも絞る
    if "採用" in dfc.columns:
        dfc = dfc[dfc["採用"]]

    if bet_type is not None and "bet_type" in dfc.columns:
        dfc = dfc[dfc["bet_type"] == bet_type]

    return dfc.reset_index(drop=True)


def apply_buy_conditions(
    df: pd.DataFrame,
    race_level: str,
    cond_win: pd.DataFrame,
    cond_plc: pd.DataFrame,
) -> pd.DataFrame:
    """df（馬単位）に 単勝/複勝 の条件判定列（単勝_条件/単勝_条件説明/複勝_条件/複勝_条件説明）を付与する
    （index_view/home/recommendations共通）。

    判定は core.strategy_engine.judge に委譲する（ルールは複数の候補特徴量
    [利益度上昇値, 人気乖離, cv, 合格数区分, 偏差値合格数区分] のうち1〜2個の組み合わせで
    表現されるため、馬ごとにこれらの生の値を渡す）。

    「人気乖離」は常に推定人気（レース前に分かる）ベースの値を指す。確定人気・単オッズ・
    複勝オッズはレース結果ファイル由来でレース後にしか確定しないため、判定には使わない。
    """
    out = df.copy()

    if not race_level:
        out["単勝_条件"] = ""
        out["単勝_条件説明"] = ""
        out["複勝_条件"] = ""
        out["複勝_条件説明"] = ""
        return out

    cw = cond_win[cond_win["レースレベル"].astype(str) == str(race_level)] if not cond_win.empty else pd.DataFrame()
    cp = cond_plc[cond_plc["レースレベル"].astype(str) == str(race_level)] if not cond_plc.empty else pd.DataFrame()

    def _feature_values(row: pd.Series) -> dict:
        return {
            "利益度上昇値": row.get("利益度上昇値"),
            "人気乖離": row.get("推定人気乖離"),
            "cv": row.get("cv"),
            "合格数区分": row.get("合格数区分"),
            "偏差値合格数区分": row.get("偏差値合格数区分"),
        }

    res_w = out.apply(lambda r: judge(_feature_values(r), cw), axis=1)
    out["単勝_条件"] = res_w.apply(lambda t: t[0])
    out["単勝_条件説明"] = res_w.apply(lambda t: t[1])

    res_p = out.apply(lambda r: judge(_feature_values(r), cp), axis=1)
    out["複勝_条件"] = res_p.apply(lambda t: t[0])
    out["複勝_条件説明"] = res_p.apply(lambda t: t[1])

    return out


def race_badge_from_horses(
    df_with_judgement: pd.DataFrame,
    win_col: str = "単勝_条件",
    plc_col: str = "複勝_条件",
) -> str:
    """
    home用：馬単位判定結果からレースバッジを決める（index_viewと整合）
    ルール：
      - 同一馬が 単勝✅ かつ 複勝✅ → ✅
      - 単勝✅ が1頭以上 → 🅰️
      - 複勝✅ が1頭以上 → 🅱️
      - 条件付き（単勝/複勝） → ☑️
      - それ以外（なし） → ""（表示なし）
    """
    if df_with_judgement is None or df_with_judgement.empty:
        return ""
    if win_col not in df_with_judgement.columns or plc_col not in df_with_judgement.columns:
        return ""

    both_same = (
        (df_with_judgement[win_col] == "✅") &
        (df_with_judgement[plc_col] == "✅")
    ).any()
    if both_same:
        return " ✅"

    has_conditional = (
        (df_with_judgement[win_col] == "△") |
        (df_with_judgement[plc_col] == "△")
    ).any()
    if has_conditional:
        return " ☑️"

    has_win = (df_with_judgement[win_col] == "✅").any()
    has_plc = (df_with_judgement[plc_col] == "✅").any()

    if has_win:
        return " 🅰️"
    if has_plc:
        return " 🅱️"
    return ""
