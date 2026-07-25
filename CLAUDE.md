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

**Step 0 — Fetch the day's race/entry data via JV-Link** (in the sibling 32bit project, see "JV-Link Integration" below):
```bash
cd ../妙味度指数_jvlink
.venv\Scripts\python.exe fetch_race_data.py --sid UNKNOWN --from <YYYYMMDD>000000 --option 2
# Output: 妙味度指数_jvlink/data_out/{ra,se}.csv (this repo's preprocessing.py reads these)
```
Also keep `data_out/horse_race_history.csv` reasonably fresh (used for 前走間隔/前走距離 — see below):
```bash
.venv\Scripts\python.exe build_horse_history.py --sid UNKNOWN   # incremental, resumes from history_state.json
```

**Step 1 — Preprocess race data** (JV-Link `data_out/` for entries + still-required TARGET manual exports at `C:/TFJV/TXT/data/` for the two fields JV-Data can't supply — see "JV-Link Integration" below):
```bash
python preprocessing.py
# Prompts: kaisai_date (e.g. 20250601), data_pattern (1=確定後, 2=確定前— only affects which peds_data columns are read)
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

`scripts/export_recommendations_to_target.py` (independent script, run any time after `scripts/auto_extract_buy_conditions.py`) writes three more marks into the *same* `C:/TFJV/target_marks_out/YYYY/YYYYMMDD/` folder, using the day's `prof_result` + `data/buy_conditions_rules.csv` (単勝/複勝/馬連軸流し, same judgment logic as `pages/recommendations.py`): 馬印3=単勝買いシグナル (◎=合致/△=一部未確定), 馬印8=複勝買いシグナル (同様), 馬印9=馬連(軸流し) (軸=軸馬/相=推奨相手). These mark numbers were free in `prof_index_calculation.py` at the time this was built — **verify they aren't already assigned to something else in your own TARGET setup before registering them**, and if they are, change `MARK_WIN`/`MARK_PLACE`/`MARK_NAGASHI` at the top of the script. Requires the same one-time TARGET-side registration (環境設定 → 馬印設定) pointing at this folder for marks 3/8/9.

### JV-Link Integration (data source for preprocessing.py)

`../妙味度指数_jvlink/` (sibling project, **32bit Python only** — JV-Link/`JVDTLab.JVLink` is a 32bit-only COM component, confirmed via registry: only registered under `WOW6432Node`) fetches race/entry data directly from JRA-VAN Data Lab via JV-Link, replacing the old TARGET-manual-CSV-export path for most fields. See that project's own README.md for full setup. Its only job is "JV-Link → normalized CSV"; this repo (64bit, pandas/streamlit) does all analysis.

- `fetch_race_data.py --sid UNKNOWN --from <date> --option {1,2,3,4}` → `data_out/{ra,se,hr_*,o1_*,o2_*}.csv`. `preprocessing.py` reads `ra.csv`/`se.csv` for the target date's race/entry info (場所,R,クラス,種別,距離,馬場状態,馬番,馬名,年齢,騎手名,調教師名 — all derived from RA/SE fields, see `_classify_class`/`_classify_track_type`/`_classify_baba_jotai` in `preprocessing.py`).
- `build_horse_history.py --sid UNKNOWN` → `data_out/horse_race_history.csv` (血統登録番号,馬名,開催年,開催月日,競馬場コード,開催回,開催日目,レース番号,距離), built from a bulk RA/SE pull and resumable via `data_out/history_state.json`'s saved `last_file_timestamp`. JV-Data's SE record has no "previous race" field at all — it only records that race's own result — so `core/horse_history.py` (`load_horse_race_history`, `build_zensou_features`) reconstructs 前走間隔/前走距離/前-2走前間隔/2-3走前間隔 by walking each horse's own race history backwards from the target date. Current backfill starts 2023-01-01; a horse whose 2nd/3rd-most-recent start falls before that has NaN there (rare edge case, ~0.5% of horses in validation — see below).
- `sid="UNKNOWN"` is fine for personal use and returns full real data (RA/SE/HR/O1-O6 etc.) — **not** a sample-only mode. (An early debugging session mistakenly concluded otherwise; the real cause was a client bug — see "known JV-Link gotchas" below.)

**Two fields still come from TARGET's manual exports** (`C:/TFJV/TXT/data/base_data/`, `C:/TFJV/TXT/data/peds_data/`) because JV-Data has no equivalent:
- **レースレベル** — TARGET's own proprietary per-race tier computation (Lv1–5), not present anywhere in raw JV-Data and not documented/derivable from it. Always at raw column 11 of `base_data_YYYYMMDD.csv` regardless of date or data_pattern (only `馬名`/`騎手名` etc. column positions shift; see "Column Layout Changes"). `preprocessing.py` still reads `base_data` for just this one column, merged onto the JV-Link-derived entries on (場所,R,馬番).
- **種牡馬名** — needs JV-Data's UM (競走馬マスタ) master file, explicitly deferred as "Phase 2" scope when `妙味度指数_jvlink` was built; `preprocessing.py` still reads `peds_data_YYYYMMDD.csv` for this one column (date/data_pattern-gated column indices, unchanged from before).

**Known JV-Link gotchas** (see `妙味度指数_jvlink/jvlink_client.py` docstrings for the fixes): `JVOpen` needs all 6 positional args (the last 3 are `[out]` params — pass `0, 0, ""` as placeholders) and returns a 4-tuple. `JVRead` returns `(return_code, buff, size, filename)` — **`return_code == -1` means "this file ended, keep reading" (not EOF!), and `0` means true EOF** (the interface spec has this exact wording; a client that treats -1 as terminal will silently only ever see the first file's records — this was the root cause of an earlier "only JG/H1 records come back" scare, not an sid restriction). `buff` is already cp932-decoded by JV-Link's COM layer, not a raw 1-byte-per-char string — re-encode with `.encode("cp932")` to get the original bytes. Always call `client.wait_for_download(result.download_count)` before reading (per spec, reading before download completion "may cause unexpected errors").

**Validation**: `preprocessing.py`'s JV-Link-sourced output was diffed field-by-field against a known-good TARGET-only `preprocessed_data_20260718.csv` (468 horses) — クラス/種別/距離/レースレベル/年齢/騎手名/調教師名/距離区分/回り/道悪判定/種牡馬名 all matched 100%; 臨戦過程/距離変遷 matched 99.4%/99.6% (the few misses are the 2023-01-01 backfill-window edge case above, not a logic bug).

### CK (出走別着度数) — running-style tendency and course/distance aptitude

`record_parsers.py`'s `CK_SPEC` covers only the horse-level portion (項番1-88, bytes 1-1384) of the 6870-byte CK record — the jockey/trainer/owner/breeder blocks that follow are out of scope. CK comes from a **different dataspec than RA/SE**: use `--dataspec SNPN` (not `SNAP`, which was retired 2023-08-08 for a producer-code-width change — `SNAP` silently stops returning data after 2023-07-31). `fetch_race_data.py --dataspec SNPN --from 20230808000000 --option 4` (chunk by ~6-month ranges — a single 3-year pull reliably crashes the CLI subprocess, cause not fully diagnosed, workaround is chunking) → `data_out/ck.csv` (436 columns: cumulative 着回数 by track/条件/distance-bracket/venue×surface, plus 脚質傾向 = JRA-VAN's own precomputed 逃げ/先行/差し/追込 counts). Unlike `horse_race_history.csv`, CK rows are already cumulative "as of entry time" snapshots — no need to walk history yourself, just fetch the target date's CK the same way as RA/SE (option=2 for 今週データ works, confirmed against real not-yet-run races).

`core/ck_features.py` derives three candidate features from `ck.csv`, attached in `preprocessing.py` right after `entries` is built (before the base_pre column trim) and merged on `CK_KEY_FIELDS` (RACE_KEY + 血統登録番号):
- **脚質傾向** — categorical, argmax of the 4 CK counts (NaN if the horse has zero prior starts).
- **コース複勝率** — numeric, 1-3着/合計 from the CK block matching this race's exact 場所+種別 (e.g. 東京芝・着回数).
- **距離帯複勝率** — numeric, same but for the 種別+距離 bracket (障害 has no distance-bracket blocks, so always NaN there).

These three are registered in `core/bet_tables.py`'s `CANDIDATE_FEATURE_COLS`, `scripts/auto_extract_buy_conditions.py`'s `CATEGORICAL_FEATURES` (脚質傾向 only — the two rate features are left continuous/qcut-binned), `core/loaders.py`'s `load_preprocessed()` column allowlist, and `buy_condition_logic.py`'s `_feature_values()`. **Historical mining won't pick these up until historical `data_out/ra.csv`/`se.csv`/`ck.csv` are backfilled and every past race day's `preprocessed_data_*.csv` is regenerated through the new `preprocessing.py`** — this integration was validated only against the current day's card so far (2026-07-18: 脚質傾向 populated for 440/468 horses, コース複勝率 267/468, 距離帯複勝率 325/468 — the rest are legitimate first-time-at-this-course/distance NaNs, not bugs).

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
2. `data/YYYY/YYYYMMDD/preprocessed_data_*.csv` → provides `レースレベル` (Lv1–Lv5) per race, and `推定人気`（推定人気ランク）used for the pre-race `人気乖離` feature. `推定人気`/`人気ランク` (and several timing/evaluation columns) come from a third-party site, smartrc.jp — not TARGET, not JV-Link — scraped via `scripts/fetch_smartrc.py` (Playwright) into `data/smartrc/smartrc_YYYYMMDD.csv`, then merged onto `preprocessed_data_*.csv` in place by `scripts/merge_smartrc_to_preprocessed.py` (run after Step 1, before Step 2/4). This is a same-day pre-race estimate, independent of both TARGET and JV-Link.
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
- Raw TFJV source CSVs (base_data/peds_data/return_data): `shift-jis` or `cp932`
- `../妙味度指数_jvlink/data_out/*.csv` (ra/se/horse_race_history etc.): `utf-8-sig`

### Column Layout Changes

`preprocessing.py` only reads raw TARGET column indices for the two fields still sourced from TARGET (see "JV-Link Integration" above):
- `レースレベル` (base_data raw column 11) — constant across all dates/data_patterns, no branching needed.
- `種牡馬名` (peds_data) — still has a date-gated branch: column indices differ for dates between 20250215–20251109 vs. all other dates, and additionally depend on `data_pattern` (1=確定後 vs 2=確定前). When modifying this extraction, always update all four branches (2 date ranges × 2 data_patterns).

Everything else (場所,R,クラス,種別,距離,馬場状態,馬番,馬名,年齢,騎手名,調教師名) is derived from JV-Link's `ra.csv`/`se.csv` by column *name*, not raw index, so it isn't affected by TARGET's export format changes at all.
