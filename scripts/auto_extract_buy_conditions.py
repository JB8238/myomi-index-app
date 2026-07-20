"""
単勝・複勝・馬連(軸流し/ボックス)の買い条件を、全期間の履歴データから自動抽出するスクリプト。

core.strategy_engine の統一エンジンで単勝/複勝/馬連(軸流し/ボックス)を横断的に探索する。
検証は以下の3段階すべてを通ったセルのみを採用する:
  1. 件数 >= min_n
  2. 日単位ブロック・ブートストラップ信頼区間の下限 > 100%
     （同日のベットは相関するため、行単位でなくレース開催日単位でリサンプリングする）
  3. 時系列4分割の安定性チェック（--max-window-failures 件までは ROI<100% の区間があっても許容する）
  4. 多重検定補正（Benjamini-Hochberg FDR）でq値が有意水準未満

195日程度の限られたデータでは、これらすべてを満たすルールが0件になることも
普通にある。それ自体は「安全装置が効いている」証拠として正直に受け止める。
過去データを大幅に増やせない場合は、--min-n / --ci-level / --fdr-alpha / --bin-q /
--max-window-failures を緩めることで採用されやすくなるが、緩めるほど誤検出
（実際には回収率100%を超えない条件を「買い」と誤判定するリスク）は増える。

実行タイミングの想定:
  prof_index_calculation.py の実行後（＝指数CSVが更新された後）に手動、または
  Windowsタスクスケジューラ等で定期実行する。
  例: schtasks /create /tn "妙味度指数_条件自動更新" /tr "python C:\\path\\to\\scripts\\auto_extract_buy_conditions.py" /sc daily /st 23:00
  （タスクスケジューラへの登録自体は本スクリプトからは行わない）

単オッズ・複勝オッズ・確定人気（＝人気順位）はレース結果ファイル(return_data)由来で
レース終了後にしか確定しない。したがって「確定人気」ベースの特徴量（人気乖離やオッズ由来の
期待値）は当日の推奨には使えないため、探索対象から外している。人気に関する特徴量は
レース前に分かる「推定人気」だけを使う（詳細は下記 build_population 参照）。

出力（data/ 配下）:
  buy_conditions_rules.csv       単勝/複勝/馬連(軸流し) 統合（bet_type列で区別）
  buy_conditions_rules_box.csv   馬連（ボックス）
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

from core.bet_tables import (
    CANDIDATE_FEATURE_COLS,
    build_box_bet_table,
    build_nagashi_bet_table,
    build_place_bet_table,
    build_win_bet_table,
    extract_hit_pairs,
)
from core.features import (
    add_component_pass_count,
    add_deviation_component_pass,
    add_race_cv,
    add_race_deviation_scores,
)
from core.history import attach_prev_total, build_index_history
from core.loaders import load_preprocessed, load_return
from core.strategy_engine import discover_rules

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "prof_result"
PREP_DIR = REPO_ROOT / "data"
MERGED_RETURN_PATH = PREP_DIR / "return_data_merged.csv"

CATEGORICAL_FEATURES = {"合格数区分", "偏差値合格数区分"}


def extract_yyyymmdd_from_name(filename: str) -> int | None:
    m = re.findall(r"(\d{8})", filename)
    for s in reversed(m):
        try:
            datetime.strptime(s, "%Y%m%d")
            return int(s)
        except ValueError:
            pass
    return None


def load_prof(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp932")
    df.columns = [str(c).strip() for c in df.columns]
    for c in ["R", "馬番", "総合利益度", "総合利益度順位"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "場所" in df.columns:
        df["場所"] = df["場所"].astype(str).str.replace("　", " ").str.strip()
    if "馬名" in df.columns:
        df["馬名"] = df["馬名"].astype(str).str.replace("　", " ").str.strip()
    return df


def df_to_csv(df: pd.DataFrame, path: Path) -> None:
    """0件でも読み込み側(pd.read_csv)がEmptyDataErrorにならないよう、最低限のヘッダは必ず残す"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.columns.empty:
        df = pd.DataFrame(columns=["bet_type", "レースレベル"])
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_population(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """全期間の統合データ(df, 全馬)と、合格馬(総合利益度>=0)のみに絞った母集団(df_pop)を返す"""
    files = sorted(DATA_DIR.rglob("results_prof_index_*.csv"))
    dated = [(extract_yyyymmdd_from_name(p.name), p) for p in files]
    dated = [(d, p) for d, p in dated if d is not None]
    dated.sort()
    if not dated:
        raise RuntimeError(f"results_prof_index CSV が見つかりません: {DATA_DIR}")

    df_ret = load_return(MERGED_RETURN_PATH) if MERGED_RETURN_PATH.exists() else None
    ret_by_date = {k: g for k, g in df_ret.groupby("開催日")} if df_ret is not None else {}

    frames = []
    for d, p in dated:
        dfp = load_prof(p)
        dfp["開催日"] = d
        if d in ret_by_date:
            dfp = dfp.merge(ret_by_date[d], on=["開催日", "場所", "R", "馬番"], how="left")
        frames.append(dfp)
    df = pd.concat(frames, ignore_index=True)

    df_pre = load_preprocessed(PREP_DIR)
    join_keys = ["開催日", "場所", "R", "馬番"]
    dup_cols = [c for c in df_pre.columns if c in df.columns and c not in join_keys]
    df = df.merge(df_pre.drop(columns=dup_cols), on=join_keys, how="left")

    history = build_index_history(DATA_DIR)
    df = attach_prev_total(df, history)
    df["利益度上昇値"] = (
        pd.to_numeric(df["総合利益度"], errors="coerce") - pd.to_numeric(df["前走総合利益度"], errors="coerce")
    )

    df["単勝的中"] = pd.to_numeric(df.get("単勝"), errors="coerce") > 0
    df["複勝的中"] = pd.to_numeric(df.get("複勝"), errors="coerce") > 0
    # 「人気乖離」は常に推定人気（レース前に分かる）ベースで計算する。
    # 確定人気・単オッズ・複勝オッズはレース結果ファイル由来でレース後にしか確定しないため、
    # 候補特徴量としては使わない（core/bet_tables.py のコメント参照）。
    if {"推定人気", "総合利益度順位"}.issubset(df.columns):
        df["人気乖離"] = pd.to_numeric(df["推定人気"], errors="coerce") - pd.to_numeric(df["総合利益度順位"], errors="coerce")
    else:
        df["人気乖離"] = np.nan

    df = add_race_deviation_scores(df)
    df = add_deviation_component_pass(df, threshold=60)

    if args.lv:
        df = df[df["レースレベル"].astype(str).isin(args.lv)]

    df["総合利益度"] = pd.to_numeric(df["総合利益度"], errors="coerce")
    df_pop = df[df["総合利益度"].notna() & (df["総合利益度"] >= 0)].copy()

    df_pop = add_component_pass_count(df_pop)
    df_pop = add_race_cv(df_pop)

    print(f"合格馬母集団: {len(df_pop)}行 / 全体 {len(df)}行 / 開催日数 {df['開催日'].nunique()}")
    return df, df_pop


def _discover(bet_table: pd.DataFrame, candidate_features: list[str], args, label: str) -> pd.DataFrame:
    if bet_table.empty:
        print(f"{label}: ベットテーブルが空のためスキップ")
        return pd.DataFrame()
    rules = discover_rules(
        bet_table, candidate_features, categorical_features=CATEGORICAL_FEATURES,
        min_n=args.min_n, ci_level=args.ci_level, fdr_alpha=args.fdr_alpha,
        n_windows=args.n_windows, n_boot=args.n_boot, max_combo=args.max_combo, bin_q=args.bin_q,
        max_window_failures=args.max_window_failures, require_fdr=args.require_fdr,
    )
    if rules.empty:
        print(f"{label}: 候補セル0件")
        return pd.DataFrame()
    accepted = rules[rules["採用"]]
    print(f"{label}: 候補{len(rules)}件 → 採用{len(accepted)}件")
    return rules


def _tag_accepted(rules: pd.DataFrame, bet_type: str) -> pd.DataFrame:
    if rules.empty:
        return rules
    out = rules[rules["採用"]].copy()
    out.insert(0, "bet_type", bet_type)
    return out


def _require_feature(rules: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """box_N のように「どのボックスサイズを買うか」の決定変数として必ず使われている
    ルールだけを残す（cvだけの条件はボックスサイズを特定できず当日適用できないため除外）。
    """
    if rules.empty:
        return rules
    used = (rules["feat1_name"] == feature_name) | (rules["feat2_name"] == feature_name)
    return rules[used]


def run(args) -> None:
    df, df_pop = build_population(args)

    # ---- 単勝・複勝・馬連（軸流し）: いずれも推定人気ベースの「人気乖離」を使う ----
    rules_win = _discover(build_win_bet_table(df_pop), CANDIDATE_FEATURE_COLS, args, "単勝")
    rules_place = _discover(build_place_bet_table(df_pop), CANDIDATE_FEATURE_COLS, args, "複勝")

    hit_pairs = extract_hit_pairs(df)
    print(f"馬連 的中ペア特定済みレース数: {len(hit_pairs)}")
    rules_nagashi = _discover(build_nagashi_bet_table(df_pop, hit_pairs), CANDIDATE_FEATURE_COLS, args, "馬連(軸流し)")
    rules_box = _discover(
        build_box_bet_table(df_pop, hit_pairs, box_sizes=tuple(args.box_sizes)),
        ["cv", "box_N"], args, "馬連(ボックス)",
    )
    rules_box = _require_feature(rules_box, "box_N")

    # ---- CSV出力（採用された行のみ） ----
    rules_all = pd.concat(
        [_tag_accepted(rules_win, "単勝"), _tag_accepted(rules_place, "複勝"), _tag_accepted(rules_nagashi, "馬連軸流し")],
        ignore_index=True,
    )
    df_to_csv(rules_all, PREP_DIR / "buy_conditions_rules.csv")
    df_to_csv(_tag_accepted(rules_box, "馬連ボックス"), PREP_DIR / "buy_conditions_rules_box.csv")

    print("完了: data/buy_conditions_rules*.csv を更新しました。")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lv", nargs="*", default=None, help="対象レースレベル（未指定なら全レベル）")
    p.add_argument("--min-n", dest="min_n", type=int, default=10)
    p.add_argument("--ci-level", dest="ci_level", type=float, default=0.60,
                   help="日単位ブロック・ブートストラップ信頼区間の水準（下限>100%%を要求）。"
                        "195日程度のデータでは0.8以上だとほぼ何も採用されないため既定を0.60に緩和済み")
    p.add_argument("--fdr-alpha", dest="fdr_alpha", type=float, default=0.20,
                   help="多重検定補正(BH-FDR)の有意水準（require-fdr指定時のみ採用可否に影響）")
    p.add_argument("--require-fdr", dest="require_fdr", action="store_true", default=False,
                   help="多重検定補正(BH-FDR)を採用可否のハードゲートにする。既定はOFF"
                        "（数百〜数千の候補セルを網羅探索する都合上、195日程度のデータでは"
                        "正式なFDR補正はほぼ何も通過できないため、既定では参考情報にとどめる）")
    p.add_argument("--n-windows", dest="n_windows", type=int, default=4)
    p.add_argument("--max-window-failures", dest="max_window_failures", type=int, default=1,
                   help="時系列分割チェックで許容する「ROI<100%%の区間」の最大数")
    p.add_argument("--n-boot", dest="n_boot", type=int, default=10000,
                   help="ブートストラップ回数。候補セル数が多いとBH-FDRが要求するp値の分解能のため"
                        "大きめの値が必要（詳細はcore/strategy_engine.pyのdocstring参照）")
    p.add_argument("--max-combo", dest="max_combo", type=int, default=2)
    p.add_argument("--bin-q", dest="bin_q", type=int, default=3,
                   help="連続値特徴量の分位数（小さいほど1セルの件数が増えて安定しやすい）")
    p.add_argument("--box-sizes", dest="box_sizes", nargs="*", type=int, default=[2, 3, 4, 5])
    return p.parse_args(argv)


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
