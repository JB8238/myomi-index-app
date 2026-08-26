"""
本日の推奨（単勝/複勝の買いシグナル、馬連(軸流し)の軸馬・推奨相手印）を
TARGET frontier JVの馬印としてエクスポートする独立スクリプト。

pages/recommendations.py と同じデータ・同じ判定ロジック（当日のprof_result +
data/buy_conditions_rules.csv の採用ルール）を使い、結果を
prof_index_calculation.py と同じ馬印エクスポート規約
（馬名,ワークデータ のヘッダなしCSV、cp932、
 C:/TFJV/target_marks_out/{year}/{kaisai_date}/ 配下）で書き出す。

書き出す馬印（★TARGET側で下記番号への割り当てを一度だけ設定する必要がある★
　　環境設定 → 馬印設定 で「work_for_mark{N}_%Y%M%D.csv」をこのフォルダから
　　読み込むよう登録する。3・8・9は本アプリの既存スクリプト
　　（prof_index_calculation.pyの1,2,4,5,6,7）で使っていない空き番号だが、
　　他の用途で既に使っている場合は書き換えてから使うこと）:
  馬印3: 単勝買いシグナル（◎=条件合致 / △=一部特徴量が未確定 / 空欄=対象外）
  馬印8: 複勝買いシグナル（同上）
  馬印9: 馬連(軸流し)（軸=このレースの軸馬 / 相=条件に合致した推奨相手 / 空欄=対象外）

買い条件が1件も見つからない当日は、全馬印が空欄のファイルが出力される
（正常な結果。無理に何かを表示することはしない）。

使い方:
  python scripts/export_recommendations_to_target.py              # 最新のprof_resultを使用
  python scripts/export_recommendations_to_target.py --date 20260719
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# scripts/ から直接実行された場合でも core/ をimportできるようにする
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buy_condition_logic import load_buy_conditions, apply_buy_conditions
from core.features import add_component_pass_count, add_race_cv_local, add_race_deviation_scores, add_deviation_component_pass
from core.history import find_prev_total, build_prof_history
from core.loaders import load_smartrc_from_preprocessed, load_preprocessed_for_race
from core.strategy_engine import judge

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "prof_result"
PREP_DIR = REPO_ROOT / "data"
MERGED_RETURN_PATH = PREP_DIR / "return_data_merged.csv"
RULES_PATH = PREP_DIR / "buy_conditions_rules.csv"

TARGET_MARKS_ROOT = Path("C:/TFJV/target_marks_out")
MARK_WIN = 3
MARK_PLACE = 8
MARK_NAGASHI = 9


def extract_yyyymmdd_from_name(filename: str) -> str | None:
    m = re.findall(r"(\d{8})", filename)
    for s in reversed(m):
        try:
            datetime.strptime(s, "%Y%m%d")
            return s
        except ValueError:
            pass
    return None


def list_prof_files() -> list[Path]:
    return sorted([p for p in DATA_DIR.rglob("*.csv") if p.is_file()])


def pick_prof_file(date_str: str | None) -> Path:
    dated = [(extract_yyyymmdd_from_name(p.name), p) for p in list_prof_files()]
    dated = [(d, p) for d, p in dated if d is not None]
    if date_str:
        matches = sorted(p for d, p in dated if d == date_str)
        if not matches:
            raise RuntimeError(f"{date_str} のprof_resultが見つかりません")
        return matches[-1]
    if not dated:
        raise RuntimeError(f"results_prof_index CSV が見つかりません: {DATA_DIR}")
    dated.sort()
    return dated[-1][1]


def load_race_level_map(prep_root: Path, target_date: str) -> dict:
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


def build_today_df(selected_file: Path, kaisai_date: str) -> pd.DataFrame:
    """pages/recommendations.py と同じ当日データ準備（前走上昇値・推定人気乖離・偏差値など）"""
    df = pd.read_csv(selected_file, encoding="cp932")

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
    df_smartrc = load_smartrc_from_preprocessed(PREP_DIR, int(kaisai_date))
    if not df_smartrc.empty:
        df = df.merge(df_smartrc, on=["場所", "R", "馬番"], how="left")

    for _col in ["脚質傾向", "コース複勝率", "距離帯複勝率"]:
        if _col in df.columns:
            df = df.drop(columns=[_col])
    df_ck_prep = load_preprocessed_for_race(PREP_DIR, int(kaisai_date))
    _ck_cols = [c for c in ["場所", "R", "馬番", "脚質傾向", "コース複勝率", "距離帯複勝率"] if c in df_ck_prep.columns]
    if not df_ck_prep.empty and "馬番" in _ck_cols:
        df = df.merge(
            df_ck_prep[_ck_cols].drop_duplicates(subset=["場所", "R", "馬番"]),
            on=["場所", "R", "馬番"],
            how="left",
        )

    if "総合利益度" in df.columns:
        history = build_prof_history(str(DATA_DIR))
        cur_mtime = selected_file.stat().st_mtime
        cur_file = selected_file.name
        df["前走総合利益度"] = df["馬名"].astype(str).apply(
            lambda n: find_prev_total(history, n, int(kaisai_date), cur_mtime, cur_file)
        )
        df["利益度上昇値"] = (
            pd.to_numeric(df["総合利益度"], errors="coerce") - pd.to_numeric(df["前走総合利益度"], errors="coerce")
        )

    # 人気乖離は常に推定人気（レース前に分かる）ベース。確定人気・単オッズ等は使わない。
    if "総合利益度順位" in df.columns:
        df["総合利益度順位"] = pd.to_numeric(df["総合利益度順位"], errors="coerce")
        if "推定人気" in df.columns:
            df["推定人気"] = pd.to_numeric(df["推定人気"], errors="coerce")
        df["推定人気乖離"] = (
            df["推定人気"] - df["総合利益度順位"] if "推定人気" in df.columns else pd.Series(np.nan, index=df.index)
        )
    else:
        df["推定人気乖離"] = np.nan

    # add_race_deviation_scores は ["開催日","場所","R"] でgroupbyするため、
    # return_data とのマージ結果（該当日がまだ return_data_merged.csv に無いと
    # 全行NaNになりうる）に依存させず、常にこの開催日で確定させる。
    df["開催日"] = int(kaisai_date)

    df = add_race_deviation_scores(df)
    df = add_deviation_component_pass(df, threshold=60)
    return df


def _feature_values(row: pd.Series) -> dict:
    # buy_condition_logic.apply_buy_conditions内の同名関数と同じ8特徴量を渡す
    # （馬連の判定はapply_buy_conditionsを経由しないためここで独自に組み立てる必要がある。
    # 以前は脚質傾向・コース複勝率・距離帯複勝率の3つが抜けており、これらを参照する
    # 馬連軸流しルールが常に判定保留(△)になっていた）。
    return {
        "利益度上昇値": row.get("利益度上昇値"),
        "人気乖離": row.get("推定人気乖離"),
        "cv": row.get("cv"),
        "合格数区分": row.get("合格数区分"),
        "偏差値合格数区分": row.get("偏差値合格数区分"),
        "脚質傾向": row.get("脚質傾向"),
        "コース複勝率": row.get("コース複勝率"),
        "距離帯複勝率": row.get("距離帯複勝率"),
    }


_WIN_PLACE_SYMBOL = {"✅": "◎", "△": "△"}


def compute_marks(df: pd.DataFrame, level_map: dict, cond_win: pd.DataFrame, cond_plc: pd.DataFrame, cond_qn: pd.DataFrame) -> pd.DataFrame:
    """戻り値: 当日の全馬について 馬名,M3,M8,M9 を持つDataFrame（対象外は空文字）"""
    m3_map: dict[str, str] = {}
    m8_map: dict[str, str] = {}
    m9_map: dict[str, str] = {}

    for (place, r), g in df.groupby(["場所", "R"]):
        lv = level_map.get((place, int(r)))

        g_base = g.copy()
        if "総合利益度" in g_base.columns:
            g_base["総合利益度"] = pd.to_numeric(g_base["総合利益度"], errors="coerce")
            g_base = g_base[g_base["総合利益度"].notna() & (g_base["総合利益度"] >= 0)]
        if g_base.empty or not lv:
            continue

        g_base = add_component_pass_count(g_base)
        g_base = add_race_cv_local(g_base)

        if not cond_win.empty or not cond_plc.empty:
            judged = apply_buy_conditions(g_base, lv, cond_win, cond_plc)
            for _, row in judged.iterrows():
                name = row.get("馬名")
                if name is None:
                    continue
                w = _WIN_PLACE_SYMBOL.get(row.get("単勝_条件"), "")
                p = _WIN_PLACE_SYMBOL.get(row.get("複勝_条件"), "")
                if w:
                    m3_map[name] = w
                if p:
                    m8_map[name] = p

        if not cond_qn.empty and "総合利益度順位" in g_base.columns:
            cond_qn_lv = cond_qn[cond_qn["レースレベル"].astype(str) == str(lv)]
            if not cond_qn_lv.empty:
                rank = pd.to_numeric(g_base["総合利益度順位"], errors="coerce")
                axis_cand = g_base[rank == 1]
                if not axis_cand.empty:
                    axis_row = axis_cand.iloc[0]
                    axis_name = axis_row.get("馬名")
                    axis_num = axis_row.get("馬番")
                    partners = g_base[g_base["馬番"] != axis_num]
                    matched = []
                    for _, prow in partners.iterrows():
                        status, *_ = judge(_feature_values(prow), cond_qn_lv)
                        if status == "✅":
                            matched.append(prow.get("馬名"))
                    if matched:
                        m9_map[axis_name] = "軸"
                        for pname in matched:
                            m9_map[pname] = "相"

    names = df["馬名"].dropna().astype(str).unique().tolist() if "馬名" in df.columns else []
    out = pd.DataFrame({"馬名": names})
    out["M3"] = out["馬名"].map(m3_map).fillna("")
    out["M8"] = out["馬名"].map(m8_map).fillna("")
    out["M9"] = out["馬名"].map(m9_map).fillna("")
    return out


def export_marks(marks_df: pd.DataFrame, kaisai_date: str) -> None:
    year = kaisai_date[:4]
    target_dir = TARGET_MARKS_ROOT / year / kaisai_date
    target_dir.mkdir(parents=True, exist_ok=True)

    for col, mark_no in [("M3", MARK_WIN), ("M8", MARK_PLACE), ("M9", MARK_NAGASHI)]:
        out = marks_df[["馬名", col]].rename(columns={col: "ワークデータ"})
        out.to_csv(
            target_dir / f"work_for_mark{mark_no}_{kaisai_date}.csv",
            index=False, header=False, encoding="cp932",
        )
    print(f"書き出し先: {target_dir}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", dest="date", default=None, help="対象開催日 YYYYMMDD（未指定なら最新のprof_result）")
    args = parser.parse_args(argv)

    selected_file = pick_prof_file(args.date)
    kaisai_date = extract_yyyymmdd_from_name(selected_file.name)
    print(f"対象ファイル: {selected_file.name}（開催日: {kaisai_date}）")

    df = build_today_df(selected_file, kaisai_date)
    level_map = load_race_level_map(PREP_DIR, kaisai_date)

    rules_mtime = RULES_PATH.stat().st_mtime if RULES_PATH.exists() else 0.0
    cond_win = load_buy_conditions(str(RULES_PATH), rules_mtime, bet_type="単勝")
    cond_plc = load_buy_conditions(str(RULES_PATH), rules_mtime, bet_type="複勝")
    cond_qn = load_buy_conditions(str(RULES_PATH), rules_mtime, bet_type="馬連軸流し")

    if cond_win.empty and cond_plc.empty and cond_qn.empty:
        print("採用された買い条件が見つかりません（scripts/auto_extract_buy_conditions.py を先に実行してください）。"
              "空欄の馬印ファイルを出力します。")

    marks_df = compute_marks(df, level_map, cond_win, cond_plc, cond_qn)
    n_win = (marks_df["M3"] != "").sum()
    n_place = (marks_df["M8"] != "").sum()
    n_nagashi = (marks_df["M9"] != "").sum()
    print(f"馬印付与数: 単勝(M3)={n_win} / 複勝(M8)={n_place} / 馬連軸流し(M9)={n_nagashi} / 全{len(marks_df)}頭")

    export_marks(marks_df, kaisai_date)
    print("完了。TARGET側で馬印3・8・9がこのフォルダを読み込むよう設定されていれば、次回読み込み時に反映されます。")


if __name__ == "__main__":
    main()
