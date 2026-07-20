"""条件探索・検証エンジン（単勝/複勝/馬連の全券種で共通利用）。

統一ベットテーブル（core.bet_tables参照: 開催日,場所,R,cost,return,<候補特徴量...>）
を受け取り、レースレベル(±全レベル込み)ごとに候補特徴量1〜2個の組み合わせでセルを作り、
以下の3段階の検証をすべて通ったセルだけを「採用ルール」として返す。

1. 件数 >= min_n
2. 日単位ブロック・ブートストラップ信頼区間の下限 > 100%
   （同日のベットは相関するため、行単位でなく日単位でリサンプリングする）
3. 時系列4分割の安定性チェック（max_window_failures件までは ROI<100% の区間があっても許容）
4. 多重検定補正（Benjamini-Hochberg FDR）でq値が有意水準未満（既定では参考情報のみ、下記参照）

195日程度の限られたデータでは、これらすべてを満たすルールが0件になることも
普通にあり得る。それ自体は「安全装置が効いている」証拠として正直に受け止める。

※ n_boot（ブートストラップ回数）は、BH-FDRが要求するp値の分解能を確保するために
  重要。同時にテストする候補セル数をnとすると、最良のセルが生き残るには最低でも
  p値が概ね fdr_alpha/n 程度まで小さく測定できる必要があり、そのためには
  n_boot が概ね n/fdr_alpha 程度（候補が千を超えるなら n_boot も万単位）は必要。
  n_bootが小さすぎると、真に良い条件があってもFDR補正を理論上決して通過できない
  （測定可能な最小p値が 1/(n_boot+1) で頭打ちになるため）。

※ require_fdr=False（既定）: レースレベル×特徴量の全組み合わせを網羅的に探索すると
  候補セル数が容易に千を超え、真に有効な条件でもBH-FDRを正式に通過するには
  p値が1/1000以下程度必要になる。195日程度のデータでそこまで小さいp値が出ることは
  ほぼ無く、FDRを必須条件にすると事実上すべて不採用になる。そのため既定では
  「多重検定OK」列は出力・参考情報として残しつつ、採用可否には使わない
  （＝件数・日次ブートストラップ信頼区間・時系列安定性の3つで判定する）。
  データ量が大きく増えるか、探索する特徴量組み合わせを絞り込んだ場合は
  require_fdr=True にして本来の多重検定補正を有効化することを推奨する。
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from core.binning import apply_edges, edges_to_bounds_df, fit_qcut_edges, in_interval


# =========================================================
# 時系列ブロック分割
# =========================================================
def compute_window_boundaries(all_dates, n_windows: int = 4) -> list[tuple[int, int]]:
    dates = sorted(set(int(d) for d in pd.Series(all_dates).dropna().tolist()))
    if not dates:
        return []
    n = len(dates)
    n_windows = max(1, min(n_windows, n))
    bounds = []
    for i in range(n_windows):
        lo_idx = int(np.floor(i * n / n_windows))
        hi_idx = int(np.floor((i + 1) * n / n_windows)) - 1
        hi_idx = max(hi_idx, lo_idx)
        bounds.append((dates[lo_idx], dates[min(hi_idx, n - 1)]))
    return bounds


def multi_window_stability(
    bet_subset: pd.DataFrame,
    window_boundaries: list[tuple[int, int]],
    date_col: str = "開催日",
    min_roi: float = 100.0,
    min_window_n: int = 3,
    max_failures: int = 1,
) -> tuple[bool, list]:
    """データが十分にある区間のうち、ROI<min_roiとなる区間が max_failures 件以下で
    あることを要求する（195日程度の限られたデータでは、1区間だけノイズで沈む程度で
    ルール全体を却下すると厳しすぎるため、既定で1区間分の逸脱は許容する）。
    どの区間にも十分なデータが無い場合は「安定性を確認できない」としてFalseにする。
    """
    window_rois = []
    supported = 0
    d = pd.to_numeric(bet_subset[date_col], errors="coerce")
    for lo, hi in window_boundaries:
        w = bet_subset[(d >= lo) & (d <= hi)]
        if len(w) < min_window_n:
            window_rois.append(None)
            continue
        supported += 1
        cost_sum = w["cost"].sum()
        roi = float(w["return"].sum() / cost_sum * 100.0) if cost_sum else None
        window_rois.append(roi)
    if supported == 0:
        return False, window_rois
    failures = sum(1 for roi in window_rois if roi is not None and roi < min_roi)
    ok = failures <= max_failures
    return ok, window_rois


# =========================================================
# 日単位ブロック・ブートストラップ
# =========================================================
def bootstrap_ci_by_day(
    bet_subset: pd.DataFrame,
    date_col: str = "開催日",
    n_boot: int = 2000,
    ci_level: float = 0.90,
    rng: np.random.Generator | None = None,
) -> dict:
    day_agg = bet_subset.groupby(date_col, observed=True).agg(cost=("cost", "sum"), ret=("return", "sum"))
    n_days = len(day_agg)
    if n_days == 0:
        return {"n_days": 0, "point_roi": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": 1.0}

    cost_by_day = day_agg["cost"].to_numpy()
    ret_by_day = day_agg["ret"].to_numpy()
    total_cost = cost_by_day.sum()
    point_roi = float(ret_by_day.sum() / total_cost * 100.0) if total_cost else float("nan")

    if n_days < 5:
        # 日数が少なすぎるとブートストラップが意味を持たないため、信頼区間なし(=不採用扱い)にする
        return {"n_days": n_days, "point_roi": point_roi, "ci_low": float("nan"), "ci_high": float("nan"), "p_value": 1.0}

    if rng is None:
        rng = np.random.default_rng(42)
    idx = rng.integers(0, n_days, size=(n_boot, n_days))
    boot_cost = cost_by_day[idx].sum(axis=1)
    boot_ret = ret_by_day[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_roi = np.where(boot_cost > 0, boot_ret / boot_cost * 100.0, np.nan)
    boot_roi = boot_roi[~np.isnan(boot_roi)]
    if boot_roi.size == 0:
        return {"n_days": n_days, "point_roi": point_roi, "ci_low": float("nan"), "ci_high": float("nan"), "p_value": 1.0}

    lo_pct = (1 - ci_level) / 2 * 100
    hi_pct = (1 + ci_level) / 2 * 100
    ci_low, ci_high = np.percentile(boot_roi, [lo_pct, hi_pct])
    p_value = float((1 + np.sum(boot_roi <= 100.0)) / (1 + boot_roi.size))
    return {
        "n_days": n_days, "point_roi": point_roi,
        "ci_low": float(ci_low), "ci_high": float(ci_high), "p_value": p_value,
    }


# =========================================================
# 多重検定補正（Benjamini-Hochberg FDR）
# =========================================================
def bh_fdr_correct(p_values, alpha: float = 0.10) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool)
    p_filled = np.where(np.isnan(p), 1.0, p)
    order = np.argsort(p_filled)
    ranked = p_filled[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    reject_sorted = np.zeros(n, dtype=bool)
    if passed.any():
        max_rank = int(np.max(np.where(passed)[0]))
        reject_sorted[: max_rank + 1] = True
    out = np.zeros(n, dtype=bool)
    out[order] = reject_sorted
    return out


# =========================================================
# 特徴量ラベル付け
# =========================================================
def _label_feature(subset: pd.DataFrame, col: str, is_categorical: bool, bin_q: int):
    """戻り値: (labels: Series, bounds_df or None, feature_type)"""
    if is_categorical:
        return subset[col].astype(str), None, "categorical"
    edges = fit_qcut_edges(subset[col], q=bin_q)
    if edges.size == 0:
        return pd.Series(np.nan, index=subset.index), None, "continuous"
    labels = apply_edges(subset[col], edges, label_prefix=f"{col}_")
    bounds = edges_to_bounds_df(edges, label_prefix=f"{col}_")
    return labels, bounds, "continuous"


# =========================================================
# メイン: ルール探索
# =========================================================
def discover_rules(
    bet_table: pd.DataFrame,
    candidate_features: list[str],
    categorical_features: set[str] | None = None,
    group_col: str = "レースレベル",
    min_n: int = 10,
    ci_level: float = 0.60,
    fdr_alpha: float = 0.20,
    n_windows: int = 4,
    n_boot: int = 10000,
    max_combo: int = 2,
    bin_q: int = 3,
    date_col: str = "開催日",
    min_window_n: int = 3,
    max_window_failures: int = 1,
    require_fdr: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    if bet_table is None or bet_table.empty:
        return pd.DataFrame()

    categorical_features = categorical_features or set()
    window_boundaries = compute_window_boundaries(bet_table[date_col], n_windows=n_windows)
    rng = np.random.default_rng(seed)

    groups: list[tuple[str, pd.DataFrame]] = []
    if group_col in bet_table.columns:
        for v in sorted(bet_table[group_col].dropna().astype(str).unique().tolist()):
            groups.append((v, bet_table[bet_table[group_col].astype(str) == v]))
    groups.append(("ALL", bet_table))

    # ---- 第1パス: 件数・点推定ROIのみで候補セルを列挙（軽量） ----
    raw_cells = []
    for group_label, subset in groups:
        if subset.empty:
            continue

        feat_info = {}
        for col in candidate_features:
            if col not in subset.columns:
                continue
            labels, bounds, ftype = _label_feature(subset, col, col in categorical_features, bin_q)
            if labels.notna().sum() == 0:
                continue
            feat_info[col] = {"labels": labels, "bounds": bounds, "type": ftype}

        feat_names = list(feat_info.keys())
        for size in range(1, min(max_combo, len(feat_names)) + 1):
            for combo in combinations(feat_names, size):
                key_df = subset[["cost", "return", date_col]].copy()
                for i, c in enumerate(combo):
                    key_df[f"_k{i}"] = feat_info[c]["labels"]
                key_cols = [f"_k{i}" for i in range(len(combo))]
                valid = key_df.dropna(subset=key_cols)
                if valid.empty:
                    continue

                stats = (
                    valid.groupby(key_cols, observed=True)
                    .agg(n=("cost", "count"), cost_sum=("cost", "sum"), return_sum=("return", "sum"))
                    .reset_index()
                )
                stats = stats[stats["n"] >= min_n]
                if stats.empty:
                    continue
                stats["roi"] = stats["return_sum"] / stats["cost_sum"] * 100.0

                for _, srow in stats.iterrows():
                    mask = pd.Series(True, index=valid.index)
                    for i in range(len(combo)):
                        mask &= (valid[f"_k{i}"] == srow[f"_k{i}"])
                    raw_cells.append({
                        "group": group_label,
                        "combo": combo,
                        "labels": tuple(srow[f"_k{i}"] for i in range(len(combo))),
                        "feat_info": {c: feat_info[c] for c in combo},
                        "n": int(srow["n"]),
                        "roi": float(srow["roi"]),
                        "rows": valid.loc[mask, ["cost", "return", date_col]],
                    })

    if not raw_cells:
        return pd.DataFrame()

    # ---- 第2パス: ブートストラップ信頼区間・時系列安定性チェック ----
    for cell in raw_cells:
        stat = bootstrap_ci_by_day(cell["rows"], date_col=date_col, n_boot=n_boot, ci_level=ci_level, rng=rng)
        cell.update(stat)
        stable, window_rois = multi_window_stability(
            cell["rows"], window_boundaries, date_col=date_col, min_roi=100.0,
            min_window_n=min_window_n, max_failures=max_window_failures,
        )
        cell["window_stable"] = stable
        cell["window_rois"] = window_rois

    # ---- 多重検定補正 ----
    reject_flags = bh_fdr_correct([c["p_value"] for c in raw_cells], alpha=fdr_alpha)
    for cell, flag in zip(raw_cells, reject_flags):
        cell["fdr_reject"] = bool(flag)

    # ---- 結果テーブル化 ----
    out_rows = []
    for cell in raw_cells:
        ci_low = cell.get("ci_low", float("nan"))
        accepted = bool(
            (not np.isnan(ci_low)) and ci_low > 100.0
            and cell.get("window_stable", False)
            and (cell.get("fdr_reject", False) or not require_fdr)
        )
        row = {
            "レースレベル": cell["group"],
            "件数": cell["n"],
            "ROI": cell["roi"],
            "CI_low": cell.get("ci_low"),
            "CI_high": cell.get("ci_high"),
            "p_value": cell.get("p_value"),
            "多重検定OK": cell.get("fdr_reject"),
            "期間安定OK": cell.get("window_stable"),
            "採用": accepted,
        }
        combo = cell["combo"]
        for slot_i, slot in enumerate(["feat1", "feat2"]):
            if slot_i >= len(combo):
                row[f"{slot}_name"] = ""
                row[f"{slot}_type"] = ""
                row[f"{slot}_value"] = np.nan
                row[f"{slot}_low"] = np.nan
                row[f"{slot}_high"] = np.nan
                row[f"{slot}_include_lowest"] = np.nan
                continue
            fname = combo[slot_i]
            info = cell["feat_info"][fname]
            label = cell["labels"][slot_i]
            row[f"{slot}_name"] = fname
            row[f"{slot}_type"] = info["type"]
            row[f"{slot}_value"] = label
            if info["type"] == "categorical":
                row[f"{slot}_low"] = np.nan
                row[f"{slot}_high"] = np.nan
                row[f"{slot}_include_lowest"] = np.nan
            else:
                b = info["bounds"]
                brow = b[b["label"] == label]
                if not brow.empty:
                    row[f"{slot}_low"] = float(brow.iloc[0]["low"])
                    row[f"{slot}_high"] = float(brow.iloc[0]["high"])
                    row[f"{slot}_include_lowest"] = bool(brow.iloc[0]["include_lowest"])
                else:
                    row[f"{slot}_low"] = np.nan
                    row[f"{slot}_high"] = np.nan
                    row[f"{slot}_include_lowest"] = np.nan
        out_rows.append(row)

    return pd.DataFrame(out_rows)


# =========================================================
# 当日適用
# =========================================================
def _match_one(row: pd.Series, prefix: str, feature_values: dict):
    fname = row.get(f"{prefix}_name")
    if fname is None or fname == "" or (isinstance(fname, float) and pd.isna(fname)):
        return True  # この特徴量スロットは未使用
    if fname not in feature_values:
        return None  # 不明（当日データに該当列がない）
    val = feature_values[fname]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None  # 不明（欠損）
    if row.get(f"{prefix}_type") == "categorical":
        return str(val) == str(row.get(f"{prefix}_value"))
    return in_interval(val, row.get(f"{prefix}_low"), row.get(f"{prefix}_high"), bool(row.get(f"{prefix}_include_lowest")))


def judge(feature_values: dict, rules_group: pd.DataFrame):
    """
    feature_values: {特徴量名: 生の値, ...}（判定したい馬・相手・レースの実際の値）
    rules_group: 対象グループ(レースレベル等)に絞り込み済みのルール表（discover_rulesの出力で"採用"==Trueの行）
    戻り値: (status, reason, roi, n)  status: "✅"（条件合致） / "△"（一部未確定） / ""（非該当）
    """
    if rules_group is None or rules_group.empty:
        return "", "", None, None

    candidates = []
    unknown_any = False
    for _, row in rules_group.iterrows():
        r1 = _match_one(row, "feat1", feature_values)
        r2 = _match_one(row, "feat2", feature_values)
        if r1 is None or r2 is None:
            unknown_any = True
            continue
        if r1 and r2:
            candidates.append(row)

    if candidates:
        best = max(candidates, key=lambda r: r["ROI"])
        parts = []
        for slot in ["feat1", "feat2"]:
            if best.get(f"{slot}_name"):
                parts.append(f"{best[f'{slot}_name']}={best.get(f'{slot}_value')}")
        reason = " & ".join(parts) + f"（件数={int(best['件数'])}）"
        return "✅", reason, float(best["ROI"]), int(best["件数"])

    if unknown_any:
        return "△", "一部の特徴量（オッズ等）が未確定のため判定保留", None, None

    return "", "", None, None
