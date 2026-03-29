# buy_condition_logic.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def load_buy_conditions(path_str: str, file_mtime: float) -> pd.DataFrame:
    """analysisが出力した buy_conditions_full_*.csv を読み込む（統一ローダ）"""
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()

    dfc = pd.read_csv(p, encoding="utf-8-sig")
    dfc.columns = [str(c).strip() for c in dfc.columns]

    for c in ["up_low", "up_high", "gap_low", "gap_high", "件数", "単勝ROI", "複勝ROI"]:
        if c in dfc.columns:
            dfc[c] = pd.to_numeric(dfc[c], errors="coerce")

    for c in ["up_include_lowest", "gap_include_lowest"]:
        if c in dfc.columns:
            dfc[c] = dfc[c].astype(bool)

    return dfc


def in_interval(val: float, low: float, high: float, include_lowest: bool) -> bool:
    if pd.isna(val) or pd.isna(low) or pd.isna(high):
        return False
    if include_lowest:
        return (val >= low) and (val <= high)
    return (val > low) and (val <= high)


def judge_row(val_up, val_gap, cond_df_lv: pd.DataFrame):
    """
    index_view と同一の判定ルール（ここが唯一の正）
    戻り値: (status, reason)
      status: "✅" / "△" / ""
    """
    if cond_df_lv is None or cond_df_lv.empty or pd.isna(val_up):
        return "", ""

    cand = cond_df_lv[
        cond_df_lv.apply(
            lambda r: in_interval(val_up, r["up_low"], r["up_high"], r["up_include_lowest"]),
            axis=1
        )
    ]
    if cand.empty:
        return "", ""

    # 人気乖離が無い（事前）なら △（条件付き）
    if pd.isna(val_gap):
        gaps = " / ".join(cand["人気乖離区分"].astype(str).unique().tolist())
        return "△", f"人気乖離が {gaps} なら買い"

    ok = cand[
        cand.apply(
            lambda r: in_interval(val_gap, r["gap_low"], r["gap_high"], r["gap_include_lowest"]),
            axis=1
        )
    ]
    if ok.empty:
        gaps = " / ".join(cand["人気乖離区分"].astype(str).unique().tolist())
        return "", f"人気乖離が {gaps} なら買い（現在は不一致）"

    # best row（説明用）
    best = ok.copy()
    roi_col = "単勝ROI" if "単勝ROI" in ok.columns else ("複勝ROI" if "複勝ROI" in ok.columns else None)
    if roi_col:
        best = best.sort_values(roi_col, ascending=False)
    r0 = best.iloc[0]
    return "✅", f"{r0['上昇値区分']} & {r0['人気乖離区分']} (件数={int(r0['件数'])})"


def apply_buy_conditions(df: pd.DataFrame, race_level: str, cond_win: pd.DataFrame, cond_plc: pd.DataFrame) -> pd.DataFrame:
    """df（馬単位）に 単勝/複勝 の条件判定列を付与する（index_view/home共通）"""
    out = df.copy()
    if not race_level:
        out["単勝_条件"] = ""
        out["単勝_条件説明"] = ""
        out["複勝_条件"] = ""
        out["複勝_条件説明"] = ""
        return out

    cw = cond_win[cond_win["レースレベル"] == race_level] if not cond_win.empty else pd.DataFrame()
    cp = cond_plc[cond_plc["レースレベル"] == race_level] if not cond_plc.empty else pd.DataFrame()

    res_w = out.apply(lambda r: judge_row(r.get("利益度上昇値"), r.get("人気乖離"), cw), axis=1)
    out["単勝_条件"] = res_w.apply(lambda t: t[0])
    out["単勝_条件説明"] = res_w.apply(lambda t: t[1])

    res_p = out.apply(lambda r: judge_row(r.get("利益度上昇値"), r.get("人気乖離"), cp), axis=1)
    out["複勝_条件"] = res_p.apply(lambda t: t[0])
    out["複勝_条件説明"] = res_p.apply(lambda t: t[1])

    return out


def race_badge_from_horses(df_with_judgement: pd.DataFrame) -> str:
    """
    home用：馬単位判定結果からレースバッジを決める（index_viewと整合）
    ルール：
      - 同一馬が 単勝✅ かつ 複勝✅ → ✅
      - 単勝✅ が1頭以上 → 🅰️
      - 複勝✅ が1頭以上 → 🅱️
      - それ以外（△のみ/なし） → ""（表示なし）
    """
    if df_with_judgement is None or df_with_judgement.empty:
        return ""

    both_same = ((df_with_judgement["単勝_条件"] == "✅") & (df_with_judgement["複勝_条件"] == "✅")).any()
    if both_same:
        return " ✅"

    has_win = (df_with_judgement["単勝_条件"] == "✅").any()
    has_plc = (df_with_judgement["複勝_条件"] == "✅").any()

    if has_win:
        return " 🅰️"
    if has_plc:
        return " 🅱️"
    return ""
