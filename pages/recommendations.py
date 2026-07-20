import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from buy_condition_logic import load_buy_conditions, apply_buy_conditions
from core.features import add_component_pass_count, add_race_cv_local, add_race_deviation_scores, add_deviation_component_pass
from core.history import find_prev_total, build_prof_history
from core.loaders import load_smartrc_from_preprocessed
from core.strategy_engine import judge

DATA_DIR = Path("prof_result")
PREP_DIR = Path("data")
MERGED_RETURN_PATH = Path("./data/return_data_merged.csv")
RULES_PATH = Path("./data/buy_conditions_rules.csv")
RULES_BOX_PATH = Path("./data/buy_conditions_rules_box.csv")

st.set_page_config(page_title="本日のおすすめレース", page_icon="🔎", layout="wide")
st.title("🔎 本日のおすすめレース")
st.caption("単勝・複勝・馬連(軸流し/ボックス)の自動抽出条件を当日レースに適用し、条件に合致した買い目を一覧表示します。")
st.divider()

st.page_link("app.py", label="← 開催レース一覧へ戻る", icon="🏠", use_container_width=True)
st.divider()


# -----------------------------------
# ファイル選択（app.pyと同じ方式）
# -----------------------------------
def extract_yyyymmdd_from_name(filename: str):
    m = re.findall(r"(\d{8})", filename)
    for s in reversed(m):
        try:
            datetime.strptime(s, "%Y%m%d")
            return s
        except ValueError:
            pass
    return None


def list_csv_files():
    return sorted([p for p in DATA_DIR.rglob("*.csv") if p.is_file()])


def pick_latest_by_filename(files):
    dated = []
    for f in files:
        d = extract_yyyymmdd_from_name(f.name)
        if d:
            dated.append((d, f))
    dated.sort()
    return dated[-1][1] if dated else None


@st.cache_data(show_spinner="📘 レースレベル (preprocessed_data) を読み込んでいます…")
def load_race_level_map(prep_root: Path, target_date: str | None) -> dict:
    if target_date is None:
        return {}
    year = target_date[:4]
    ymd_dir = prep_root / year / target_date
    if not ymd_dir.exists():
        return {}
    files = list(ymd_dir.glob("preprocessed_data_*.csv"))
    if not files:
        return {}
    path = sorted(files)[-1]
    dfp = pd.read_csv(path, encoding="utf-8")
    required = {"場所", "R", "レースレベル"}
    if not required.issubset(dfp.columns):
        return {}
    dfp = dfp.copy()
    dfp["R"] = pd.to_numeric(dfp["R"], errors="coerce")
    dfp["レースレベル"] = dfp["レースレベル"].astype(str).str.strip()
    level_map = {}
    for (place, r), g in dfp.dropna(subset=["R"]).groupby(["場所", "R"]):
        lv = g["レースレベル"].dropna()
        if not lv.empty:
            level_map[(place, int(r))] = lv.mode().iloc[0]
    return level_map


files = list_csv_files()
if not files:
    st.error("prof_result にCSVが見つかりません")
    st.stop()

files_sorted = sorted(
    files,
    key=lambda p: (extract_yyyymmdd_from_name(p.name) or "0", p.name),
    reverse=True,
)
latest = pick_latest_by_filename(files_sorted)

with st.sidebar:
    st.header("対象レース")
    selected_file = st.selectbox(
        "読み込みCSV（デフォルトは最新）",
        options=files_sorted,
        index=files_sorted.index(latest),
        format_func=lambda p: p.name,
    )

df = pd.read_csv(selected_file, encoding="cp932")
kaisai_date = extract_yyyymmdd_from_name(selected_file.name)
st.caption(f"参照ファイル: {selected_file.name} （開催日: {kaisai_date if kaisai_date else '-'}）")

if not {"場所", "R"}.issubset(df.columns):
    st.error("CSVに「場所」「R」列が必要です")
    st.stop()

# -----------------------------------
# return_data / smartrc マージ（app.pyと同じ）
# -----------------------------------
if MERGED_RETURN_PATH.exists():
    df_return = pd.read_csv(MERGED_RETURN_PATH, encoding="utf-8-sig", low_memory=False)
    df_return.columns = [str(c).strip() for c in df_return.columns]
    df_return.rename(columns={"Ｒ": "R"}, inplace=True)
    for c in ["開催日", "R", "馬番"]:
        if c in df_return.columns:
            df_return[c] = pd.to_numeric(df_return[c], errors="coerce")
    if "場所" in df_return.columns:
        df_return["場所"] = df_return["場所"].astype(str).str.replace("　", " ").str.strip()

    if "場所" in df.columns:
        df["場所"] = df["場所"].astype(str).str.replace("　", " ").str.strip()
    if "R" in df.columns:
        df["R"] = pd.to_numeric(df["R"], errors="coerce")
    if "馬番" in df.columns:
        df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")

    df_ret_day = df_return[df_return["開催日"] == int(kaisai_date)]
    df = df.merge(df_ret_day, on=["場所", "R", "馬番"], how="left", validate="m:1")

for _col in ["推定人気", "人気ランク"]:
    if _col in df.columns:
        df = df.drop(columns=[_col])
_df_smartrc_prep = load_smartrc_from_preprocessed(PREP_DIR, int(kaisai_date))
if not _df_smartrc_prep.empty:
    df = df.merge(_df_smartrc_prep, on=["場所", "R", "馬番"], how="left")

# -----------------------------------
# 前走比較・人気乖離・偏差値
# -----------------------------------
if "総合利益度" in df.columns:
    history = build_prof_history(str(DATA_DIR))
    cur_date = int(kaisai_date)
    cur_mtime = selected_file.stat().st_mtime
    cur_file = selected_file.name
    df["前走総合利益度"] = df["馬名"].astype(str).apply(
        lambda n: find_prev_total(history, n, cur_date, cur_mtime, cur_file)
    )
    df["利益度上昇値"] = (
        pd.to_numeric(df["総合利益度"], errors="coerce") - pd.to_numeric(df["前走総合利益度"], errors="coerce")
    )

# 人気乖離は常に推定人気（レース前に分かる）ベース。確定人気・単オッズ等は
# レース結果ファイル由来でレース後にしか確定しないため使わない。
if "総合利益度順位" in df.columns:
    df["総合利益度順位"] = pd.to_numeric(df["総合利益度順位"], errors="coerce")
    if "推定人気" in df.columns:
        df["推定人気"] = pd.to_numeric(df["推定人気"], errors="coerce")
    df["推定人気乖離"] = (
        df["推定人気"] - df["総合利益度順位"] if "推定人気" in df.columns else pd.Series(np.nan, index=df.index)
    )
else:
    df["推定人気乖離"] = np.nan

df = add_race_deviation_scores(df)
df = add_deviation_component_pass(df, threshold=60)

level_map = load_race_level_map(PREP_DIR, kaisai_date)

# -----------------------------------
# 買い条件CSV読み込み（bet_typeで絞り込み）
# -----------------------------------
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


_rules_mtime = _mtime(RULES_PATH)
_box_mtime = _mtime(RULES_BOX_PATH)
cond_win = load_buy_conditions(str(RULES_PATH), _rules_mtime, bet_type="単勝")
cond_plc = load_buy_conditions(str(RULES_PATH), _rules_mtime, bet_type="複勝")
cond_qn = load_buy_conditions(str(RULES_PATH), _rules_mtime, bet_type="馬連軸流し")
cond_qb = load_buy_conditions(str(RULES_BOX_PATH), _box_mtime, bet_type="馬連ボックス")

if cond_win.empty and cond_plc.empty and cond_qn.empty and cond_qb.empty:
    st.warning(
        "採用された買い条件が見つかりません。scripts/auto_extract_buy_conditions.py を実行してください"
        "（データ量や検証基準によっては、採用件数が0件になるのは正常な結果です）。"
    )

with st.expander("読み込んだ買い条件（採用ルール）の状況"):
    st.write(f"単勝: {len(cond_win)}件 / 複勝: {len(cond_plc)}件 / "
             f"馬連(軸流し): {len(cond_qn)}件 / 馬連(ボックス): {len(cond_qb)}件")

st.divider()


def _feature_values(row: pd.Series) -> dict:
    return {
        "利益度上昇値": row.get("利益度上昇値"),
        "人気乖離": row.get("推定人気乖離"),
        "cv": row.get("cv"),
        "合格数区分": row.get("合格数区分"),
        "偏差値合格数区分": row.get("偏差値合格数区分"),
    }


# -----------------------------------
# レースごとに推奨を組み立て
# -----------------------------------
recommendations = []

for (place, r), g in df.groupby(["場所", "R"]):
    r_int = int(r)
    lv = level_map.get((place, r_int))

    g_base = g.copy()
    if "総合利益度" in g_base.columns:
        g_base["総合利益度"] = pd.to_numeric(g_base["総合利益度"], errors="coerce")
        g_base = g_base[g_base["総合利益度"].notna() & (g_base["総合利益度"] >= 0)]
    if g_base.empty:
        continue

    g_base = add_component_pass_count(g_base)
    g_base = add_race_cv_local(g_base)

    # ---- 単勝・複勝 ----
    if not cond_win.empty or not cond_plc.empty:
        judged = apply_buy_conditions(g_base, lv, cond_win, cond_plc)
        for _, row in judged[judged.get("単勝_条件", "") == "✅"].iterrows():
            recommendations.append({
                "場所": place, "R": r_int, "レースレベル": lv, "券種": "単勝",
                "内容": f"{row.get('馬名', '?')}（{int(row['馬番'])}番）",
                "根拠": row.get("単勝_条件説明", ""),
            })
        for _, row in judged[judged.get("複勝_条件", "") == "✅"].iterrows():
            recommendations.append({
                "場所": place, "R": r_int, "レースレベル": lv, "券種": "複勝",
                "内容": f"{row.get('馬名', '?')}（{int(row['馬番'])}番）",
                "根拠": row.get("複勝_条件説明", ""),
            })

    # ---- 馬連（軸流し） ----
    if not cond_qn.empty and lv and "総合利益度順位" in g_base.columns:
        cond_qn_lv = cond_qn[cond_qn["レースレベル"].astype(str) == str(lv)]
        if not cond_qn_lv.empty:
            rank = pd.to_numeric(g_base["総合利益度順位"], errors="coerce")
            axis_cand = g_base[rank == 1]
            if not axis_cand.empty:
                axis_row = axis_cand.iloc[0]
                axis_num = axis_row.get("馬番")
                axis_name = axis_row.get("馬名", "?")

                partners = g_base[g_base["馬番"] != axis_num]
                for _, prow in partners.iterrows():
                    status, reason, roi, n = judge(_feature_values(prow), cond_qn_lv)
                    if status == "✅":
                        recommendations.append({
                            "場所": place, "R": r_int, "レースレベル": lv, "券種": "馬連(軸流し)",
                            "内容": f"軸:{axis_name}（{int(axis_num)}番）－相手:{prow.get('馬名', '?')}（{int(prow['馬番'])}番）",
                            "根拠": f"{reason}（ROI≈{roi:.0f}%）" if roi is not None else reason,
                        })

    # ---- 馬連（ボックス） ----
    if not cond_qb.empty and lv and "cv" in g_base.columns and "総合利益度順位" in g_base.columns:
        cond_qb_lv = cond_qb[cond_qb["レースレベル"].astype(str) == str(lv)]
        if not cond_qb_lv.empty:
            race_cv_series = pd.to_numeric(g_base["cv"], errors="coerce").dropna()
            race_cv = float(race_cv_series.iloc[0]) if not race_cv_series.empty else np.nan

            box_n_values = set()
            for slot in ["feat1", "feat2"]:
                sub = cond_qb_lv[cond_qb_lv[f"{slot}_name"] == "box_N"]
                box_n_values.update(sub[f"{slot}_value"].tolist())

            best = None
            for box_n in sorted(box_n_values, key=lambda v: str(v)):
                try:
                    box_n_int = int(box_n)
                except (TypeError, ValueError):
                    continue
                if len(g_base) < box_n_int:
                    continue
                status, reason, roi, n = judge({"cv": race_cv, "box_N": str(box_n_int)}, cond_qb_lv)
                if status == "✅" and (best is None or roi > best[1]):
                    best = (box_n_int, roi, reason, n)

            if best is not None:
                box_n_int, roi, reason, n = best
                top = g_base.sort_values("総合利益度順位").head(box_n_int)
                names = "・".join(f"{row.get('馬名', '?')}（{int(row['馬番'])}）" for _, row in top.iterrows())
                recommendations.append({
                    "場所": place, "R": r_int, "レースレベル": lv, "券種": "馬連(ボックス)",
                    "内容": names,
                    "根拠": f"{reason}（ROI≈{roi:.0f}%）",
                })

# -----------------------------------
# 表示
# -----------------------------------
if not recommendations:
    st.info("本日は条件に合致するおすすめレース・買い目が見つかりませんでした。")
else:
    rec_df = pd.DataFrame(recommendations)
    st.success(f"本日のおすすめ: {len(rec_df)}件（{rec_df['R'].nunique()}レース）")

    for place, g_place in rec_df.groupby("場所"):
        st.markdown(f"## 📍 {place}")
        for r_int, g_race in g_place.groupby("R"):
            lv = g_race["レースレベル"].iloc[0]
            st.markdown(f"**{int(r_int)}R**（{lv or '-'}）")
            st.dataframe(
                g_race[["券種", "内容", "根拠"]],
                hide_index=True,
                use_container_width=True,
            )
        st.divider()
