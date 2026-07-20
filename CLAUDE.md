# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Japanese horse racing index (妙味度指数) analysis application built with Streamlit. It computes profitability indices (利益度) for jockeys, sires, and trainers, then displays race-day metrics and betting condition recommendations.

## Running the App

```bash
# Install dependencies
pip install -r requirement.txt

# Start the Streamlit app
streamlit run app.py
```

The app runs multi-page: `app.py` is the home page (race list), with `pages/index_view.py` (per-race horse detail) and `pages/recommendations.py` (本日のおすすめレース — applies all buy-condition rule CSVs to the day's races) as sub-pages. There is no manual condition-tuning UI anymore — `pages/analysis.py` was retired; all condition discovery is done by `scripts/auto_extract_buy_conditions.py` (see below).

## Data Pipeline (run in order before using the app)

All scripts are run manually with Python — they are **not** part of the Streamlit app.

**Step 1 — Preprocess raw race data** (requires external TFJV data at `C:/TFJV/TXT/data/`):
```bash
python preprocessing.py
# Prompts: kaisai_date (e.g. 20250601), data_pattern (1=確定後, 2=確定前)
# Output: data/YYYY/YYYYMMDD/preprocessed_data_YYYYMMDD.csv
```

**Step 2 — Calculate profitability indices**:
```bash
python prof_index_calculation.py
# Prompts: kaisai_date
# Reads: list/YYYY/jockey_prof_list_YYYY.csv, sire_prof_list_YYYY.csv, trainer_prof_list_YYYY.csv
#         data/YYYY/YYYYMMDD/preprocessed_data_YYYYMMDD.csv
# Output: prof_result/YYYY/results_prof_index_YYYYMMDD.csv
#         index/YYYYMMDD/{jockey,sire,trainer}_index_YYYYMMDD.csv
#         C:/TFJV/target_marks_out/YYYY/YYYYMMDD/work_for_mark{1,2,4,5,6,7}_YYYYMMDD.csv (TARGET馬印取込用)
#         C:/TFJV/EX_DATA/妙味度指数/myomido_index_YYYYMMDD.csv (TARGET外部指数取込用、総合利益度)
```

### TARGET frontier JV Integration

`prof_index_calculation.py` writes two kinds of files consumed by TARGET frontier JV, both under `C:/TFJV/` (outside this repo, on the machine running TARGET):

- **馬印 (marks)**: `C:/TFJV/target_marks_out/YYYY/YYYYMMDD/work_for_markN_YYYYMMDD.csv` — headerless `馬名,ワークデータ` CSVs imported as TARGET's 馬印 1/2/4/5/6/7.
- **外部指数 (external index)**: `C:/TFJV/EX_DATA/妙味度指数/myomido_index_YYYYMMDD.csv` — headerless `レースID,指数` CSV (cp932), one row per horse with a non-NaN `総合利益度`. The race ID is TARGET's "第3仕様" 14-digit import format (`YYYY+MM+DD+場所コード2桁+R2桁+馬番2桁`, no 回/日次needed) built via the `PLACE_CODE` map in `prof_index_calculation.py` (JRA 10 tracks only; unmapped 場所 values are skipped with a warning).
  - One-time setup required inside TARGET (環境設定 → 外部指数の設定 → 新規追加): ファイル形式「馬単位・CSV形式」, パス・ファイル名 `C:\TFJV\EX_DATA\妙味度指数\myomido_index_%Y3%M1%D1.csv`, レースID「第3仕様(12/14桁)」, 指数順位判定「大きい方が優位」. After that one-time registration, TARGET auto-loads whatever file this script produces for the matching date.

**Step 3 — Build merged return data** (one-time / periodic refresh):
```bash
python scripts/build_return_data_merged.py
# Reads: C:/TFJV/TXT/data/return_data/YYYY/return_data_YYYYMMDD.csv
# Output: data/return_data_merged.csv
```

**Step 4 — Auto-extract buy conditions** (run after Step 2, whenever prof_result/ has new data):
```bash
python scripts/auto_extract_buy_conditions.py
# Reads: prof_result/**, data/**, data/return_data_merged.csv (all dates, non-interactive)
# Output: data/buy_conditions_rules.csv        単勝/複勝/馬連(軸流し) 統合、bet_type列で区別
#         data/buy_conditions_rules_box.csv    馬連（ボックス）
```
This is the sole condition-discovery mechanism (the old manual `pages/analysis.py` UI has been retired). It uses `core.strategy_engine.discover_rules` — a single engine shared by 単勝/複勝/馬連(軸流し)/馬連(ボックス) — to search combinations of 1–2 candidate features (利益度上昇値, 人気乖離, cv, 合格数区分, 偏差値合格数区分) per レースレベル (+ a pooled "ALL" group), and keeps only cells that pass:

`人気乖離` here is always computed from **推定人気**（推定人気 − 総合利益度順位), never from 確定人気/単オッズ/複勝オッズ — those come from the return_data (results) file and are only known after the race finishes, so they are useless for a same-day recommendation and are intentionally excluded from the candidate feature set entirely (there is no `EV_win`/`EV_place` feature anymore; `core/ev.py` was removed).
1. `件数 >= min_n`
2. a **day-clustered bootstrap confidence interval** (resample race *days*, not rows, since same-day bets are correlated) whose lower bound exceeds 100% ROI
3. a **4-block chronological stability check** (ROI >= 100% in all but `--max-window-failures` time blocks with enough data)

Benjamini-Hochberg FDR correction is also computed (`多重検定OK` column) but is **informational only by default** (`require_fdr=False` / no `--require-fdr` flag) — with ~195 race days and a combinatorial search of a few hundred to ~1400 candidate cells per bet type, a formally-corrected p-value would need to be smaller than roughly 1/1000 to survive, which essentially never happens with this much data. Treating it as a hard gate made the whole pipeline accept zero rules across every bet type; pass `--require-fdr` once there is much more history (or a narrower search) if you want the full, stricter guarantee back. Even without the FDR hard-gate, accepted rules here are "leans profitable" (day-clustered one-sided p-values around 0.05–0.2, i.e. roughly 80–95% one-sided confidence, not 99%+) rather than a certainty — this is the deliberate trade-off of loosening thresholds instead of collecting more data.

Current defaults reflect this tuning for a small (~195-day) dataset: `--min-n 10`, `--ci-level 0.60`, `--fdr-alpha 0.20` (informational), `--n-windows 4` with `--max-window-failures 1`, `--n-boot 10000` (needs to be large — see `core/strategy_engine.py` docstring — or BH-FDR can never mathematically pass anything), `--bin-q 3`. It is normal — not a bug — for 馬連 in particular to still accept zero rules even with these looser settings (e.g. the best axis+nagashi candidate found so far had ROI concentrated entirely in the most recent quarter with zero hits in the prior three, which the stability check correctly rejects as more likely a hot streak than a persistent edge). Re-run this script any time you want recommendations to reflect the latest data (e.g. after each racing day) — it always recomputes from scratch and overwrites the CSVs above. There is no scheduled-task registration wired up yet — set one up yourself (e.g. Windows Task Scheduler) if you want it to run automatically. See `--help` for all flags.

## Architecture

### Directory Layout

```
prof_result/        # Index CSVs consumed by the Streamlit app (cp932 encoding)
data/               # Preprocessed race data + merged return data + buy condition rule CSVs
  YYYY/YYYYMMDD/    # preprocessed_data_YYYYMMDD.csv (race metadata)
  buy_conditions_rules.csv       # 単勝/複勝/馬連(軸流し) 統合ルール（bet_type列で区別）
  buy_conditions_rules_box.csv   # 馬連（ボックス）ルール
  return_data_merged.csv         # Merged payout data for all dates
index/YYYYMMDD/     # Per-date index CSVs for each category
list/YYYY/          # Annual master lists (jockey/sire/trainer prof lists)
```

### Key Data Flow in the App

1. `prof_result/` CSVs → loaded by `app.py` and `pages/index_view.py`
2. `data/YYYY/YYYYMMDD/preprocessed_data_*.csv` → provides `レースレベル` (Lv1–Lv5) per race, and `推定人気`（推定人気ランク）used for the pre-race `人気乖離` feature
3. `data/return_data_merged.csv` → joined to add actual payout results (post-race only; used for backtesting and results display, never as a live judgment feature)
4. `data/buy_conditions_rules*.csv` → generated by `scripts/auto_extract_buy_conditions.py`, used by home/index_view (win+place badges) and `pages/recommendations.py` (win+place+馬連) to show betting condition recommendations

### Module Roles

- **`core/features.py`** — pure pandas transformations: component pass counts, CV (coefficient of variation), deviation scores (偏差値)
- **`core/history.py`** — builds a history table from all `prof_result/` CSVs to look up a horse's previous 総合利益度 (`build_prof_history`, `find_prev_total`)
- **`core/loaders.py`** — cached loaders for preprocessed data and return CSVs
- **`core/binning.py`** — generic "fit qcut edges on the available data, then reuse those exact numeric edges (via `pd.cut`) both for validation and for same-day application" utility (`fit_qcut_edges`, `apply_edges`, `edges_to_bounds_df`, `in_interval`). Replaces the old fixed hand-picked bin boundaries.
- **`core/bet_tables.py`** — normalizes every bet type (単勝, 複勝, 馬連軸流し, 馬連ボックス) into one shared schema: `開催日,場所,R,レースレベル,cost,return,<candidate features>`. `extract_hit_pairs` recovers actual 馬連 winning pairs from `return_data_merged.csv`. Candidate features are deliberately limited to ones knowable *before* the race (利益度上昇値, 人気乖離 via 推定人気, cv, 合格数区分, 偏差値合格数区分) — no odds/確定人気-derived features.
- **`core/strategy_engine.py`** — the condition-discovery engine, shared by all bet types via the unified bet-table schema above. `discover_rules()` searches 1–2-feature combinations per レースレベル (+ pooled "ALL"), and keeps only cells passing a day-clustered bootstrap CI and a 4-block temporal stability check (see Step 4 above for the full rationale, including why BH-FDR is informational-only by default). `judge()` applies an accepted rules table to a single horse/candidate's live feature values (status "✅"/"△"/"") and is the one judgment function used everywhere (win/place/馬連).
- **`buy_condition_logic.py`** — thin wrapper around `core.strategy_engine.judge` for the 単勝/複勝 (per-horse) case, used by `app.py`, `pages/index_view.py`, and `pages/recommendations.py`; `apply_buy_conditions()` is the authoritative entry point (always keys the `人気乖離` feature off the caller's `推定人気乖離` column). `load_buy_conditions()` reads a rules CSV and optionally filters by `bet_type`.
- **`scripts/auto_extract_buy_conditions.py`** — the only condition-discovery entry point (see Step 4 above); builds the population, builds all 4 bet tables, runs `discover_rules`, writes the rule CSVs.

### Key Domain Concepts

- **総合利益度** — overall profitability index (horse passes threshold at >= 0). This "合格馬" population (総合利益度>=0) is the one hard prerequisite kept in the new engine; everything else (合格数区分, cv, etc.) is a searchable candidate feature rather than a fixed pre-filter.
- **利益度上昇値** — increase from previous race's 総合利益度
- **人気乖離** — difference between 推定人気 (pre-race estimated popularity rank) and 総合利益度 rank. Always estimated-popularity-based, never 確定人気/オッズ-based, because those are only known once the race is over (see Step 4 above).
- **レースレベル** — race tier (Lv1–Lv5) from preprocessing; Lv4/Lv5 are highlighted with 🔥, Lv3 with ⭐ when a 17+ index horse exists
- **Buy condition badges** (win/place, on home/index_view): ✅ (same horse hits win+place), 🅰️ (win only), 🅱️ (place only), ☑️ (conditional)
- **馬連 (quinella) strategies** — two independently backtested patterns, not a single condition space like win/place: axis+nagashi (軸=総合利益度順位1位かつ総合利益度>=0固定、相手はcore.strategy_engineの候補特徴量で条件化) and top-N box (総合利益度順位の上位N頭を総当たり、box_Nと混戦度cvで条件化)

### Encoding Notes

- `prof_result/` CSVs: `cp932`
- `data/return_data_merged.csv`: `utf-8-sig`
- `data/preprocessed_data_*.csv`: `utf-8`
- Raw TFJV source CSVs: `shift-jis`

### Column Layout Changes

`preprocessing.py` has a date-gated branch: column indices differ for dates between 20250215–20251109 vs. all other dates. When modifying column extraction logic, always update both branches.
