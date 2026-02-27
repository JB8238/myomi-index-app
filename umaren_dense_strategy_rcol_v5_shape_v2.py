#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
all_race_results_2.csv（R列あり）対応：単勝/複勝 + 馬連（組み合わせ）探索

✅ 本スクリプトで出来ること
- 既存の「topk（指数順位）× 複勝オッズ帯 × 人気レンジ × diff_pop × 荒れLv（レース印１）」の枠組みで
  単勝/複勝の成績を集計
- 同じフィルタ条件で、馬連（UMAREN）を以下の買い方で集計
  - umaren_box   : 抽出馬のボックス（全組み合わせ）
  - umaren_axis1 : 軸=抽出馬内の指数1位（score順位1位）、相手=同一レース内の「印あり & 単勝オッズ<=上限」へ流す（買い目を増やす改造）

馬連データ仕様（ユーザー要件）
- 「馬連」列は 1着/2着馬の行に数値が入る想定
- レースの馬連配当は「そのレース内の馬連列の最大値」で代表

重要：全角数字対応
- インプットの「着順」等が '１','２' のような全角数字の場合があるため、
  数値化前に全角→半角へ正規化します（これをしないと馬連の的中判定が全て0になります）。

出力
- analysis_output/umaren_dense_summary_rcol.csv : 全条件
- analysis_output/umaren_dense_top_rcol.csv     : 上位のみ（見やすい）

例）
python umaren_dense_strategy_rcol.py --input_csv "./all_race_results_2.csv" --min_bets 200
python umaren_dense_strategy_rcol.py --input_csv "./all_race_results_2.csv" --rlevels "3" --min_bets 200
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

BET_AMOUNT = 100

# 全角→半角（数字のみ）
FW_TO_HW = str.maketrans("０１２３４５６７８９", "0123456789")


def normalize_numeric_token(x: object) -> str:
    """
    数値列の文字列を頑健に正規化。
    - 全角数字→半角
    - "１着" / "2着" / " 3 " のような混在から先頭の数値(整数/小数)を抽出
    - 数値が取れない場合は空文字
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip().translate(FW_TO_HW)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else ""


def to_halfwidth_digits(x: object) -> str:
    # 互換のため残置（古い処理）。基本は normalize_numeric_token を使う。
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip().translate(FW_TO_HW)


def read_csv_robust(path: Path) -> pd.DataFrame:
    encs = ["utf-8-sig", "utf-8", "cp932", "shift-jis", "euc-jp"]
    last_err = None
    for enc in encs:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err  # type: ignore[misc]


def normalize_date(x: object) -> str:
    """'2026. 2. 1' のような表記揺れを '2026-02-01' に寄せる"""
    s = str(x).strip()
    m = re.search(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})", s)
    if not m:
        return s
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"{y:04d}-{mo:02d}-{d:02d}"


def normalize_rlevel(x: object) -> Optional[int]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    m = re.search(r"Lv\s*([1-5])", s, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def ensure_numeric_from_text(df: pd.DataFrame, cols: List[str]) -> None:
    """
    文字列/全角混在の可能性がある列を、抽出→数値化。
    例：'１着' / '2着' / '3 ' / '取消' などの混在に対応。
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].apply(normalize_numeric_token), errors="coerce")


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out

def parse_float_list(s: str) -> List[Optional[float]]:
    """
    カンマ区切りの浮動小数候補をパース。空文字なら [None]（フィルタ無し）。
    例: "0,1.5,2" -> [0.0, 1.5, 2.0]
    """
    ss = str(s).strip()
    if not ss:
        return [None]
    out: List[Optional[float]] = []
    for p in ss.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(float(p))
    return out if out else [None]



def parse_ranges(s: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
        if not m:
            raise ValueError(f"pop_ranges format error: {part} (expected like 3-9)")
        a, b = int(m.group(1)), int(m.group(2))
        if b < a:
            a, b = b, a
        ranges.append((a, b))
    return ranges


def round1(x: float) -> float:
    return float(f"{x:.1f}")



def filter_picked_by_shape(
    picked: pd.DataFrame,
    race_meta: Dict[str, Dict[str, object]],
    gap12_min: Optional[float],
    std_max: Optional[float],
) -> pd.DataFrame:
    """
    race_meta に保存した指数の形（gap/分散）を使って、picked をレース単位で絞り込む。
    - gap12_min: score1-score2 の下限（Noneなら無視）
    - std_max: score上位Nのstdの上限（Noneなら無視）
    """
    if picked.empty:
        return picked
    if gap12_min is None and std_max is None:
        return picked

    ok_races = []
    for rid, meta in race_meta.items():
        if gap12_min is not None:
            g = meta.get("gap12", None)
            if g is None or float(g) < float(gap12_min):
                continue
        if std_max is not None:
            s = meta.get("std_topn", None)
            if s is None or float(s) > float(std_max):
                continue
        ok_races.append(rid)

    if not ok_races:
        return picked.iloc[0:0].copy()

    return picked[picked["race_id"].isin(set(ok_races))].copy()

def calc_roi_horse_bets(picked: pd.DataFrame) -> Dict[str, float]:
    """単勝/複勝（馬単位）"""
    bets = int(len(picked))
    if bets == 0:
        return {
            "bets": 0,
            "win_roi": float("nan"),
            "win_hit": float("nan"),
            "place_roi": float("nan"),
            "place_hit": float("nan"),
        }

    win_payout = pd.to_numeric(picked.get("単勝配当", 0), errors="coerce").fillna(0).sum()
    place_payout = pd.to_numeric(picked.get("複勝配当", 0), errors="coerce").fillna(0).sum()
    win_hit = (pd.to_numeric(picked.get("単勝配当", 0), errors="coerce").fillna(0) > 0).mean()
    place_hit = (pd.to_numeric(picked.get("複勝配当", 0), errors="coerce").fillna(0) > 0).mean()

    return {
        "bets": bets,
        "win_roi": float(win_payout / (bets * BET_AMOUNT) * 100),
        "win_hit": float(win_hit),
        "place_roi": float(place_payout / (bets * BET_AMOUNT) * 100),
        "place_hit": float(place_hit),
    }


def umaren_box_bets_hit(selected_nums: List[int], actual_pair: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    """
    ボックス：選んだ馬番集合に actual_pair の2頭が含まれていれば的中。
    """
    n = len(set(selected_nums))
    if n < 2:
        return 0, 0
    bets = n * (n - 1) // 2
    if actual_pair is None:
        return bets, 0
    s = set(selected_nums)
    hit = 1 if (actual_pair[0] in s and actual_pair[1] in s) else 0
    return bets, hit


def umaren_axis1_bets_hit(
    selected: pd.DataFrame,
    marked_nums: List[int],
    actual_pair: Optional[Tuple[int, int]],
) -> Tuple[int, int]:
    """
    軸1頭流し（拡張版）：
    - 軸：selected（条件を満たした馬）のうち score順位1位
    - 相手：同一レース内の「印あり & 単勝オッズ<=上限」の馬（--mark_col/--mark_set/--win_odds_max）
    """
    # 軸を作るには、条件該当馬が最低1頭必要
    if len(selected) < 1:
        return 0, 0

    sel = selected.dropna(subset=["馬番_num", "rank_score"]).copy()
    sel["rank_score_num"] = pd.to_numeric(sel["rank_score"], errors="coerce")
    sel = sel.dropna(subset=["rank_score_num"]).sort_values(["rank_score_num", "馬番_num"])
    if len(sel) < 1:
        return 0, 0

    axis = int(sel.iloc[0]["馬番_num"])

    # 相手：marked_nums から軸を除外
    others = set(int(x) for x in marked_nums if int(x) != axis)
    bets = len(others)
    if bets <= 0:
        return 0, 0

    if actual_pair is None:
        return bets, 0

    a, b = actual_pair
    hit = 1 if ((axis == a and b in others) or (axis == b and a in others)) else 0
    return bets, hit


def build_race_meta(
    df: pd.DataFrame,
    mark_col: str,
    mark_set: List[str],
    win_odds_col: str,
    win_odds_max: float,
    shape_topn: int,
) -> Dict[str, Dict[str, object]]:
    """
    race_id ごとに、実際の1-2着馬番ペアと馬連配当、そして
    「軸流しの相手候補（印あり & 単勝オッズ<=win_odds_max）」の馬番リストを事前計算。
    """
    meta: Dict[str, Dict[str, object]] = {}

    # 印セット（空文字などを除外）
    ms = set([str(x).strip() for x in mark_set if str(x).strip()])

    for rid, r in df.groupby("race_id"):
        rr = r.dropna(subset=["着順_num", "馬番_num"]).copy()
        top2 = rr[rr["着順_num"].isin([1, 2])].sort_values("着順_num")
        if len(top2) >= 2:
            a = int(top2.iloc[0]["馬番_num"])
            b = int(top2.iloc[1]["馬番_num"])
            actual_pair: Optional[Tuple[int, int]] = tuple(sorted((a, b)))
        else:
            actual_pair = None

        umaren = pd.to_numeric(r.get("馬連", 0), errors="coerce").fillna(0).max()

        # 相手候補：印あり & 単勝オッズ<=上限（単勝オッズ列が無い場合は印のみ）
        cand = r.dropna(subset=["馬番_num"]).copy()
        cand["mark"] = cand[mark_col].astype(str).str.strip()
        cand = cand[cand["mark"].isin(ms)]
        if win_odds_col:
            cand = cand.dropna(subset=[win_odds_col]).copy()
            cand = cand[cand[win_odds_col] <= float(win_odds_max)]
        marked_nums = sorted(set(cand["馬番_num"].astype(int).tolist()))

        # 指数の形（gap/分散）
        if "score_total" in r.columns:
            sc = pd.to_numeric(r["score_total"], errors="coerce").fillna(0)
        else:
            sc = (
                pd.to_numeric(r.get("馬印5", 0), errors="coerce").fillna(0)
                + pd.to_numeric(r.get("馬印6", 0), errors="coerce").fillna(0)
                + pd.to_numeric(r.get("馬印7", 0), errors="coerce").fillna(0)
            ).fillna(0)
        sc_pos = sc[sc > 0].sort_values(ascending=False).tolist()

        gap12 = None
        if len(sc_pos) >= 2:
            gap12 = float(sc_pos[0] - sc_pos[1])

        std_topn = None
        if len(sc_pos) >= 2:
            n = int(shape_topn) if int(shape_topn) > 0 else 5
            n = min(len(sc_pos), n)
            std_topn = float(np.std(sc_pos[:n], ddof=0))

        meta[rid] = {
            "actual_pair": actual_pair,
            "umaren_payout": float(umaren),
            "marked_nums": marked_nums,
            "gap12": gap12,
            "std_topn": std_topn,
        }


    return meta


def calc_umaren_for_config(picked: pd.DataFrame, race_meta: Dict[str, Dict[str, object]], mode: str) -> Tuple[int, float, int]:
    """
    mode: umaren_box / umaren_axis1
    戻り: (total_bets, roi, race_count_with_bets)
    """
    total_bets = 0
    total_payout = 0.0
    race_count = 0

    for rid, sel in picked.groupby("race_id"):
        meta = race_meta.get(rid)
        if not meta:
            continue

        actual_pair = meta["actual_pair"] if isinstance(meta.get("actual_pair"), tuple) else None
        umaren = float(meta.get("umaren_payout", 0.0))

        if mode == "umaren_box":
            nums = sel.dropna(subset=["馬番_num"])["馬番_num"].astype(int).tolist()
            bets, hit = umaren_box_bets_hit(nums, actual_pair)
        elif mode == "umaren_axis1":
            marked_nums = meta.get("marked_nums", [])
            bets, hit = umaren_axis1_bets_hit(sel, marked_nums, actual_pair)
        else:
            raise ValueError(f"unknown pair mode: {mode}")

        if bets <= 0:
            continue

        race_count += 1
        total_bets += bets
        if hit == 1:
            total_payout += umaren

    roi = (total_payout / (total_bets * BET_AMOUNT) * 100) if total_bets > 0 else float("nan")
    return int(total_bets), float(roi), int(race_count)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True, help="all_race_results_2.csv のパス")
    ap.add_argument("--output_dir", type=str, default="./analysis_output")
    ap.add_argument("--include_zero_score", action="store_true", help="馬印合計=0も母集団に含める")
    ap.add_argument("--min_bets", type=int, default=200, help="単複（馬単位）ベット数の下限")
    ap.add_argument("--top_n", type=int, default=50, help="コンソール表示の上位行数")
    ap.add_argument("--export_top", type=int, default=300, help="top.csvに出す行数")

    ap.add_argument("--topk_list", type=str, default="2,3")
    ap.add_argument("--high_max", type=float, default=3.0)
    ap.add_argument("--low_min_start", type=float, default=1.4)
    ap.add_argument("--low_min_end", type=float, default=2.2)
    ap.add_argument("--low_min_step", type=float, default=0.1)

    ap.add_argument("--pop_ranges", type=str, default="2-6,2-7,3-7,3-8")
    ap.add_argument("--diff_pop_list", type=str, default="0,1")
    ap.add_argument("--rlevels", type=str, default="1,2,3,4,5")
    ap.add_argument("--pair_modes", type=str, default="umaren_box,umaren_axis1")
    ap.add_argument("--mark_col", type=str, default="馬印4", help="軸流しの相手候補に使う印列")
    ap.add_argument("--mark_set", type=str, default="◎,○,▲,△,★,×", help="相手候補に含める印（カンマ区切り。◇は既定で除外）")
    ap.add_argument("--win_odds_col", type=str, default="", help="単勝オッズ列名（空なら列名に「単勝オッズ」を含む列を自動探索）")
    ap.add_argument("--win_odds_max", type=float, default=100.0, help="軸流しの相手候補に含める単勝オッズ上限（例：100）")
    # 指数の形（gap/分散）フィルタ
    ap.add_argument("--shape_topn", type=int, default=5, help="指数分散（std）を計算する上位N（score>0の馬を降順で）")
    ap.add_argument("--gap12_mins", type=str, default="", help="gap12（score1-score2）の下限候補（カンマ区切り）。空ならフィルタ無し")
    ap.add_argument("--std_max_list", type=str, default="", help="score上位Nの標準偏差（std）の上限候補（カンマ区切り）。空ならフィルタ無し")
    args = ap.parse_args()

    df = read_csv_robust(Path(args.input_csv))

    required = [
        "日付(yyyy.mm.dd)", "場所", "Ｒ", "レース名", "クラス名", "レース印１",
        "頭数", "馬番", "人気", "着順",
        "複勝オッズ下限", "複勝オッズ上限",
        "単勝配当", "複勝配当",
        "馬印4",
        "馬印5", "馬印6", "馬印7",
        "馬連",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"必要列が見つかりません: {missing}")

    # 追加チェック（相手候補に使う印列）
    if args.mark_col not in df.columns:
        raise ValueError(f"--mark_col={args.mark_col} がインプットに存在しません。列名を確認してください。")

    # 単勝オッズ列の解決（空なら自動探索）
    win_odds_col = args.win_odds_col.strip()
    if not win_odds_col:
        cand = [c for c in df.columns if "単勝オッズ" in str(c)]
        win_odds_col = cand[0] if cand else ""
    if win_odds_col and win_odds_col not in df.columns:
        raise ValueError(f"--win_odds_col={win_odds_col} がインプットに存在しません。列名を確認してください。")
    df["date"] = df["日付(yyyy.mm.dd)"].apply(normalize_date)
    df["rlevel"] = df["レース印１"].apply(normalize_rlevel)
    # ユーザー方針：rlevel欠損は除外
    df = df[df["rlevel"].notna()].copy()

    # 文字列/全角混在の可能性が高い列は半角化してから数値化
    ensure_numeric_from_text(
        df,
        ["Ｒ", "頭数", "馬番", "人気", "着順", "複勝オッズ下限", "複勝オッズ上限", "単勝配当", "複勝配当", "馬連", "馬印5", "馬印6", "馬印7"],
    )

    # 単勝オッズ列がある場合は数値化（相手候補フィルタに使用）
    if win_odds_col:
        ensure_numeric_from_text(df, [win_odds_col])

    # 指数合計（馬印5-7）。欠損は0扱い（score>0 で有効馬を選別）
    df["score_total"] = (
        pd.to_numeric(df.get("馬印5", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("馬印6", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("馬印7", 0), errors="coerce").fillna(0)
    )

    # 馬印合計
    df["馬印5"] = df["馬印5"].fillna(0)
    df["馬印6"] = df["馬印6"].fillna(0)
    df["馬印7"] = df["馬印7"].fillna(0)
    df["score"] = df["馬印5"] + df["馬印6"] + df["馬印7"]

    if not args.include_zero_score:
        df = df[df["score"] > 0].copy()

    # race_id（R列があるので一意）
    df["R_str"] = df["Ｒ"].fillna(-1).astype(int).astype(str)
    df["race_id"] = df["date"].astype(str) + "_" + df["場所"].astype(str).str.strip() + "_R" + df["R_str"]

    # 数値列（下流で使いやすいよう、明示列を作る）
    df["馬番_num"] = df["馬番"]
    df["人気_num"] = df["人気"]
    df["着順_num"] = df["着順"]

    # ランク
    df["rank_score"] = df.groupby("race_id")["score"].rank(ascending=False, method="min")
    df["rank_pop"] = df.groupby("race_id")["人気_num"].rank(ascending=True, method="min")
    df["diff_pop"] = df["rank_pop"] - df["rank_score"]

    # race_meta 事前計算（相手候補を含む）
    mark_set = [x.strip() for x in str(args.mark_set).split(",") if x.strip()]
    race_meta = build_race_meta(df, args.mark_col, mark_set, win_odds_col, float(args.win_odds_max), int(args.shape_topn))

    # grids
    topk_list = parse_int_list(args.topk_list)
    pop_ranges = parse_ranges(args.pop_ranges)
    diff_pop_list = parse_int_list(args.diff_pop_list)
    gap12_mins = parse_float_list(args.gap12_mins)
    std_max_list = parse_float_list(args.std_max_list)
    rlevels = parse_int_list(args.rlevels)
    pair_modes = [x.strip() for x in args.pair_modes.split(",") if x.strip()]

    low_mins: List[float] = []
    x = args.low_min_start
    while x <= args.low_min_end + 1e-9:
        low_mins.append(round1(x))
        x += args.low_min_step
    high_max = round1(args.high_max)

    rows = []
    for rl in rlevels:
        base_lv = df[df["rlevel"] == rl].copy()
        if base_lv.empty:
            continue

        for topk in topk_list:
            base_topk = base_lv[base_lv["rank_score"] <= topk].copy()

            for lo in low_mins:
                base_band = base_topk[
                    (base_topk["複勝オッズ下限"] >= lo) & (base_topk["複勝オッズ上限"] <= high_max)
                ].copy()

                for pop_min, pop_max in pop_ranges:
                    base_pop = base_band[
                        (base_band["人気_num"] >= pop_min) & (base_band["人気_num"] <= pop_max)
                    ].copy()

                    for dmin in diff_pop_list:
                        for gap12_min in gap12_mins:
                            for std_max in std_max_list:
                                picked = base_pop[base_pop["diff_pop"] >= dmin].copy()
                                picked = filter_picked_by_shape(picked, race_meta, gap12_min, std_max)

                                stat = calc_roi_horse_bets(picked)

                                umaren_stats: Dict[str, object] = {}
                                for mode in pair_modes:
                                    b, roi, rc = calc_umaren_for_config(picked, race_meta, mode)
                                    umaren_stats[f"{mode}_bets"] = b
                                    umaren_stats[f"{mode}_roi"] = roi
                                    umaren_stats[f"{mode}_races"] = rc

                                rows.append(
                            {
                                "rlevel": rl,
                                "topk": topk,
                                "place_low_min": lo,
                                "place_high_max": high_max,
                                "pop_min": pop_min,
                                "pop_max": pop_max,
                                "diff_pop_min": dmin,
                                "gap12_min": gap12_min,
                                "std_topn": int(args.shape_topn),
                                "std_max": std_max,
                                **stat,
                                **umaren_stats,
                            }
                        )

    res = pd.DataFrame(rows)

    # 見やすい並び：複勝ROI→馬連ROI(box)→的中→bets
    sort_cols = ["place_roi"]
    if "umaren_box_roi" in res.columns:
        sort_cols.append("umaren_box_roi")
    sort_cols += ["place_hit", "bets"]
    view = res.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "umaren_dense_summary_rcol.csv"
    top_path = out_dir / "umaren_dense_top_rcol.csv"
    res.to_csv(summary_path, index=False, encoding="utf-8-sig")
    view.head(args.export_top).to_csv(top_path, index=False, encoding="utf-8-sig")

    # コンソール表示
    show_cols = [
        "rlevel", "topk", "place_low_min", "place_high_max", "pop_min", "pop_max", "diff_pop_min", "gap12_min", "std_max",
        "bets", "place_roi", "place_hit", "win_roi", "win_hit",
    ]
    for mode in pair_modes:
        show_cols += [f"{mode}_bets", f"{mode}_roi", f"{mode}_races"]

    shown = view[view["bets"] >= args.min_bets].copy()
    if shown.empty:
        shown = view.copy()

    print("=" * 170)
    print("【単複 + 馬連 探索（R列あり・全角着順対応） 上位】")
    print("=" * 170)
    print(shown[show_cols].head(args.top_n).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {top_path}")


if __name__ == "__main__":
    main()
