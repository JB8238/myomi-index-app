"""
scripts/fetch_smartrc.py

smartrc.jp から「推定人気」「人気ランク」および「評価順位」タブのデータを取得する。
単日取得と期間取得の両方に対応。

Usage:
    python scripts/fetch_smartrc.py          # 通常取得
    python scripts/fetch_smartrc.py --explore  # API フィールド調査モード（toku 全値を探索）

出力: data/smartrc/smartrc_YYYYMMDD.csv
  列: 開催日, 場所, R, 馬番, 馬名, 推定人気, 人気ランク,
      テン1F_過去, テン1F_過去_ランク, テン1F_前走, テン1F_前走_ランク,
      テン_T, テン_T_ランク, 上がり_T, 上がり_T_ランク,
      前走_評価, 前々走_評価, 3走前_評価, 4走前_評価, 5走前_評価

APIフィールド名（--explore で確認済み）:
  テン1F_過去    : ten1f_best   / ランク: ten1f_best_rank
  テン1F_前走    : h1_ten1f     / ランク: h1_ten1f_rank
  テン_T        : ten_has      / ランク: ten_has_rank
  上がり_T      : agari_has    / ランク: agari_has_rank
  前走_評価     : h1_fr_baba
  前々走_評価   : h2_fr_baba
  3走前_評価    : h3_fr_baba
  4走前_評価    : h4_fr_baba
  5走前_評価    : h5_fr_baba
  ※ 全toku値で同じフィールドが返るためtoku=0で取得
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd
from playwright.async_api import async_playwright

SMARTRC_BASE = "https://www.smartrc.jp/v3"
SMARTRC_URL  = SMARTRC_BASE + "/"
OUT_DIR      = Path("data/smartrc")

VENUE_CODE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}
VENUE_NAME_TO_CODE = {v: k for k, v in VENUE_CODE_MAP.items()}

# ①〜⑱ を整数に変換するマッピング
CIRCLE_NUMS = {chr(0x2460 + i): i + 1 for i in range(18)}

# ── 評価順位タブのフィールド名（--explore で確定済み） ──────────────
# 値フィールド → (値キー, ランクキー)  ※値とランクは別フィールドで返る
TIMED_FIELDS = {
    "テン1F_過去": ("ten1f_best",  "ten1f_best_rank"),
    "テン1F_前走": ("h1_ten1f",    "h1_ten1f_rank"),
    "テン_T":      ("ten_has",     "ten_has_rank"),
    "上がり_T":    ("agari_has",   "agari_has_rank"),
}

# 前走～5走前の評価フィールド（h?_fr_baba）
EVAL_FIELDS = {
    "前走_評価":   "h1_fr_baba",
    "前々走_評価": "h2_fr_baba",
    "3走前_評価":  "h3_fr_baba",
    "4走前_評価":  "h4_fr_baba",
    "5走前_評価":  "h5_fr_baba",
}


# ── ユーティリティ ───────────────────────────────────────────────

def decode_rcode(rcode: str) -> tuple | None:
    if len(rcode) != 16:
        return None
    date_str   = rcode[:8]
    venue_code = rcode[8:10]
    race_num   = int(rcode[14:16])
    venue_name = VENUE_CODE_MAP.get(venue_code)
    if venue_name is None:
        return None
    return date_str, venue_name, race_num


def get_venue_codes(kaisai_date: str) -> list[str]:
    """preprocessed_data から開催会場コードを取得する。なければ全会場を返す。"""
    year      = kaisai_date[:4]
    prep_path = Path(f"data/{year}/{kaisai_date}/preprocessed_data_{kaisai_date}.csv")
    if prep_path.exists():
        df     = pd.read_csv(prep_path, encoding="utf-8")
        venues = df["場所"].str.strip().unique().tolist() if "場所" in df.columns else []
        codes  = [VENUE_NAME_TO_CODE[v] for v in venues if v in VENUE_NAME_TO_CODE]
        if codes:
            return list(dict.fromkeys(codes))
    return list(VENUE_CODE_MAP.keys())


def api_url(endpoint: str) -> str:
    dc = int(time.time() * 1000)
    return f"{SMARTRC_BASE}/smartrc.php/{endpoint}?_dc={dc}"


async def post_api(context, endpoint: str, body: dict, csrf: str) -> dict | None:
    try:
        resp = await context.request.post(
            api_url(endpoint),
            headers={
                "Content-Type":     "application/json",
                "X-Requested-With": "XMLHttpRequest",
                "Csrftoken":        csrf,
                "Referer":          SMARTRC_URL,
                "Origin":           "https://www.smartrc.jp",
            },
            data=json.dumps(body),
        )
        if resp.status != 200:
            return None
        return await resp.json()
    except Exception as e:
        print(f"  APIエラー ({endpoint}): {e}")
        return None


# ── 評価順位タブ データパース ────────────────────────────────────

def to_float_div10(raw) -> float | None:
    """3桁整数（例: 355）を 35.5 に変換する。None や非数値は None を返す。"""
    if raw is None or raw == "":
        return None
    try:
        return int(str(raw).strip()) / 10
    except (ValueError, TypeError):
        return None


def to_int_rank(raw) -> int | None:
    """ランク整数文字列（例: "3"）を int に変換する。None や非数値は None を返す。"""
    if raw is None or raw == "":
        return None
    try:
        return int(str(raw).strip())
    except (ValueError, TypeError):
        return None


def extract_eval_fields(horse: dict) -> dict:
    """
    runners/view レスポンスの1頭分データから評価順位タブの各フィールドを抽出する。
    値とランクは別フィールドで返るため、それぞれ個別に取得する。
    """
    row: dict = {}
    for col, (val_key, rank_key) in TIMED_FIELDS.items():
        row[col]            = to_float_div10(horse.get(val_key))
        row[f"{col}_ランク"] = to_int_rank(horse.get(rank_key))

    for col, key in EVAL_FIELDS.items():
        v = horse.get(key)
        row[col] = v if v not in (None, "") else None

    return row


def detect_eval_toku(toku_data: dict[str, list]) -> str | None:
    """
    explore で取得した各 toku のサンプルデータから、
    評価順位タブと思われる toku 値を自動検出する。
    """
    eval_hints = {"ten1f_best", "ten_has", "agari_has", "h1_ustat", "h1_ten1f"}
    best_toku  = None
    best_score = 0

    for toku, fields in toku_data.items():
        score = sum(1 for f in fields if f in eval_hints)
        if score > best_score:
            best_score = score
            best_toku  = toku

    return best_toku


# ── 全フィールド値ダンプ（--dump モード用） ──────────────────────

async def dump_all_fields(ctx, csrf: str, kaisai_date: str):
    """
    runners/view の全フィールド値を1レース分出力して JSON に保存する。
    h1_rcode が存在する（前走データがある）馬のみ対象。
    評価フィールドの特定に使用する。
    """
    venue_codes = get_venue_codes(kaisai_date)
    target_rcode: str | None = None

    for venue_code in venue_codes:
        data = await post_api(ctx, "races/view", {
            "rdate": kaisai_date,
            "place": venue_code,
            "page": 1, "start": 0, "limit": 10,
        }, csrf)
        if data and data.get("success") and data.get("data"):
            # 頭数が多いレースを選ぶ（データが揃いやすい）
            races = sorted(data["data"], key=lambda r: int(r.get("entry", "0") or 0), reverse=True)
            target_rcode = races[0]["rcode"]
            vname = VENUE_CODE_MAP[venue_code]
            print(f"  対象: {vname} {decode_rcode(target_rcode)[2]}R (rcode={target_rcode})")
            break

    if not target_rcode:
        print("  レースが見つかりません")
        return

    data = await post_api(ctx, "runners/view", {
        "rcode": target_rcode,
        "toku":  "0",
        "page":  1, "start": 0, "limit": 100,
    }, csrf)
    if not data or not data.get("data"):
        print("  データ取得失敗")
        return

    # 前走データがある馬を最大5頭選ぶ
    horses_with_prev = [
        h for h in data["data"]
        if h.get("h1_rcode") and h.get("h1_rank")
    ][:5]

    if not horses_with_prev:
        print("  前走データがある馬が見つかりません")
        return

    print(f"\n  前走データあり: {len(horses_with_prev)} 頭を出力")

    dump: list[dict] = []
    for horse in horses_with_prev:
        entry = {
            "_info": {
                "uno":    horse.get("uno"),
                "hname":  horse.get("hname"),
                "h1_rank": horse.get("h1_rank"),
                "h1_f3rank": horse.get("h1_f3rank"),
                "h1_ustat":  horse.get("h1_ustat"),
                "h2_ustat":  horse.get("h2_ustat"),
            },
            "all_fields": dict(horse),
        }
        dump.append(entry)

        # コンソールに評価候補フィールドを出力
        uno  = horse.get("uno", "?")
        name = horse.get("hname", "?")
        print(f"\n  馬番{uno} {name}:")
        for i in range(1, 6):
            rcode_key = f"h{i}_rcode"
            if not horse.get(rcode_key):
                break
            fields_of_interest = {
                k: v for k, v in horse.items()
                if k.startswith(f"h{i}_") and v not in (None, "", "0", 0)
            }
            print(f"    h{i}: " + ", ".join(f"{k}={repr(v)}" for k, v in sorted(fields_of_interest.items())))

    out_path = OUT_DIR / "dump_fields.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 全フィールド値を保存しました: {out_path}")
    print("  このファイルで h1_ustat 等の実際の値を確認し、評価フィールドを特定してください。")


# ── フィールド探索（--explore モード用） ─────────────────────────

async def explore_fields(ctx, csrf: str, kaisai_date: str, page):
    """
    runners/view の全 toku 値（0〜6）のレスポンスフィールドを調査し、
    評価順位タブのフィールド名特定に役立つ情報を出力する。
    結果を data/smartrc/explore_fields.json に保存する。
    """
    venue_codes = get_venue_codes(kaisai_date)
    target_rcode: str | None = None

    for venue_code in venue_codes:
        data = await post_api(ctx, "races/view", {
            "rdate": kaisai_date,
            "place": venue_code,
            "page": 1, "start": 0, "limit": 10,
        }, csrf)
        if data and data.get("success") and data.get("data"):
            target_rcode = data["data"][0]["rcode"]
            vname = VENUE_CODE_MAP[venue_code]
            print(f"  調査対象: {vname} {decode_rcode(target_rcode)[2]}R (rcode={target_rcode})")
            break

    if not target_rcode:
        print("  レースが見つかりません")
        return

    report      = {}
    toku_fields = {}  # toku値 -> フィールド名リスト（自動検出用）

    for toku_val in ["0", "1", "2", "3", "4", "5", "6"]:
        data = await post_api(ctx, "runners/view", {
            "rcode": target_rcode,
            "toku":  toku_val,
            "page":  1, "start": 0, "limit": 100,
        }, csrf)
        key = f"toku_{toku_val}"
        if data and data.get("data"):
            first = data["data"][0]
            all_fields = sorted(first.keys())
            toku_fields[toku_val] = all_fields
            report[key] = {
                "all_keys":     all_fields,
                "total_fields": len(first),
                "sample":       dict(list(first.items())[:80]),
            }
            print(f"  runners/view toku={toku_val}: {len(first)} フィールド取得")
        else:
            report[key] = {"error": "no data"}
            print(f"  runners/view toku={toku_val}: データなし")
        await asyncio.sleep(0.5)

    # ── 評価順位タブの自動検出
    best_toku = detect_eval_toku(toku_fields)

    # ── フィールド名候補レポート
    print("\n" + "=" * 60)
    print("【評価順位フィールド候補】")
    print(f"  推定される評価順位 toku 値: {best_toku}")
    print()

    eval_keywords = {
        "テン1F 過去": ["ten", "t1f", "tlt"],
        "テン1F 前走": ["ten", "t1f", "pre"],
        "テン T":      ["ten_t", "tnt", "ttime"],
        "上がり T":    ["age_t", "agt", "ltime", "utime", "last"],
        "評価":        ["_hy", "eval", "hyoka"],
    }

    for toku_val, fields in toku_fields.items():
        if not fields:
            continue
        matches = {}
        for label, hints in eval_keywords.items():
            found = [f for f in fields if any(h in f.lower() for h in hints)]
            if found:
                matches[label] = found
        if matches:
            marker = " ← 評価順位候補" if toku_val == best_toku else ""
            print(f"  toku={toku_val}{marker}:")
            for label, found in matches.items():
                print(f"    {label}: {', '.join(found)}")
            print()

    report["_summary"] = {
        "estimated_eval_toku": best_toku,
        "toku_field_counts":   {k: len(v) for k, v in toku_fields.items()},
    }

    # ── ブラウザで評価順位タブをクリックして追加APIを検出
    extra_api_calls = []

    async def capture_eval_tab_request(response):
        if response.status != 200:
            return
        ct = response.headers.get("content-type", "")
        if "json" not in ct:
            return
        if "smartrc.php" not in response.url:
            return
        try:
            rdata = await response.json()
            extra_api_calls.append({
                "url":       response.url,
                "post_data": response.request.post_data,
                "data_keys": list(rdata.keys()) if isinstance(rdata, dict) else None,
                "item_keys": (
                    sorted(rdata["data"][0].keys())
                    if isinstance(rdata, dict) and rdata.get("data") and rdata["data"]
                    else None
                ),
                "sample": (
                    dict(list(rdata["data"][0].items())[:60])
                    if isinstance(rdata, dict) and rdata.get("data") and rdata["data"]
                    else None
                ),
            })
        except Exception:
            pass

    page.on("response", capture_eval_tab_request)

    print("ブラウザで評価順位タブをクリック中...")
    try:
        await page.goto(SMARTRC_URL)
        await page.wait_for_load_state("networkidle", timeout=20000)
        await page.wait_for_timeout(2000)

        eval_tab = page.locator("text=評価順位").first
        if await eval_tab.count() > 0:
            await eval_tab.click()
            await page.wait_for_timeout(3000)
            print("  評価順位タブをクリックしました")
        else:
            print("  ⚠ 評価順位タブが見つかりません")
            print("  手動でレースをクリックし、評価順位タブをクリックしてください（15秒）")
            await page.wait_for_timeout(15000)
    except Exception as e:
        print(f"  ブラウザ操作エラー: {e}")

    report["extra_api_on_eval_tab"] = extra_api_calls
    if extra_api_calls:
        print(f"\n  評価順位タブクリックで {len(extra_api_calls)} 件の追加APIコールを検出:")
        for call in extra_api_calls:
            print(f"    URL: {call['url']}")
            print(f"    POST: {call['post_data']}")
            if call.get("item_keys"):
                print(f"    フィールド: {', '.join(call['item_keys'][:20])}")

    # ── レポート保存
    out_path = OUT_DIR / "explore_fields.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 現在の TIMED_FIELDS / EVAL_FIELDS との整合チェック
    all_field_names: set[str] = set()
    for flist in toku_fields.values():
        all_field_names.update(flist)

    print("\n" + "=" * 60)
    print("【現在の TIMED_FIELDS / EVAL_FIELDS との整合チェック】")
    ok = True
    for col, (val_key, rank_key) in TIMED_FIELDS.items():
        v_ok = "✓" if val_key  in all_field_names else "✗ 未検出"
        r_ok = "✓" if rank_key in all_field_names else "✗ 未検出"
        print(f"  {col}: 値={val_key}({v_ok})  ランク={rank_key}({r_ok})")
        if "未検出" in v_ok or "未検出" in r_ok:
            ok = False
    for col, key in EVAL_FIELDS.items():
        found = "✓" if key in all_field_names else "✗ 未検出"
        print(f"  {col}: {key}({found})")
        if "未検出" in found:
            ok = False

    if ok:
        print("\n  ✅ 全フィールドが確認されました。通常取得を実行できます。")
    else:
        print("\n  ⚠ 未検出フィールドがあります。TIMED_FIELDS / EVAL_FIELDS を更新してください。")

    out_path = OUT_DIR / "explore_fields.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 探索結果を保存しました: {out_path}")


# ── 1開催日分のデータ取得 ────────────────────────────────────────

async def fetch_one_date(ctx, csrf: str, kaisai_date: str) -> pd.DataFrame | None:
    """
    1開催日分の全レース・全馬データを取得して DataFrame を返す。
    全toku値で同じフィールドが返るため toku=0 のみを使用する。
    """
    venue_codes = get_venue_codes(kaisai_date)

    all_rcodes: list[str] = []
    for venue_code in venue_codes:
        data = await post_api(ctx, "races/view", {
            "rdate": kaisai_date,
            "place": venue_code,
            "page": 1, "start": 0, "limit": 100,
        }, csrf)
        if not data or not data.get("success") or not data.get("data"):
            continue
        rcodes = [r["rcode"] for r in data["data"] if r.get("rcode")]
        all_rcodes.extend(rcodes)
        vname = VENUE_CODE_MAP[venue_code]
        print(f"    {vname}: {len(rcodes)} レース")

    if not all_rcodes:
        print(f"  → レースなし（スキップ）")
        return None

    eval_fields_found = False
    all_rows: list[dict] = []

    for rcode in all_rcodes:
        decoded = decode_rcode(rcode)
        if not decoded:
            continue
        date_str, venue_name, race_num = decoded

        data = await post_api(ctx, "runners/view", {
            "rcode": rcode,
            "toku":  "0",
            "page":  1, "start": 0, "limit": 100,
        }, csrf)
        if not data or not data.get("success") or not data.get("data"):
            print(f"    ✗ 取得失敗: {venue_name} {race_num}R")
            continue

        for horse in data["data"]:
            row = {
                "開催日":     int(date_str),
                "場所":       venue_name,
                "R":          race_num,
                "馬番":       int(horse["uno"]),
                "馬名":       horse.get("hname", ""),
                "推定人気":   int(horse["est_pop"]) if horse.get("est_pop") else None,
                "人気ランク": horse.get("old_pr"),
            }
            eval_row = extract_eval_fields(horse)
            row.update(eval_row)

            if not eval_fields_found and any(v is not None for v in eval_row.values()):
                eval_fields_found = True

            all_rows.append(row)

        print(f"    ✓ 取得: {venue_name} {race_num}R ({len(data['data'])}頭)")
        await asyncio.sleep(0.3)

    if not all_rows:
        return None

    if not eval_fields_found:
        print(
            "  ⚠ 評価順位フィールドが1件も取得できませんでした。\n"
            "    TIMED_FIELDS / EVAL_FIELDS のキー名を確認してください。"
        )

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["開催日", "場所", "R", "馬番"])
    df = df.sort_values(["場所", "R", "馬番"]).reset_index(drop=True)
    return df


# ── days/view から期間内の開催日リストを取得 ───────────────────────

async def get_race_dates_in_range(ctx, csrf: str, start_date: str, end_date: str) -> list[str]:
    data = await post_api(ctx, "days/view", {
        "page": 1, "start": 0, "limit": 500,
    }, csrf)
    if not data or not data.get("data"):
        return []
    dates = [
        d["rdate"] for d in data["data"]
        if d.get("rdate") and start_date <= d["rdate"] <= end_date
    ]
    return sorted(dates)


# ── ブラウザ起動 & CSRF 取得 ─────────────────────────────────────

async def launch_browser_and_get_csrf(p):
    browser = await p.chromium.launch(headless=False)
    ctx     = await browser.new_context()
    page    = await ctx.new_page()

    csrf_holder: list[str] = []

    def capture_csrf(request):
        if "smartrc.php" in request.url and not csrf_holder:
            tok = request.headers.get("csrftoken", "")
            if tok:
                csrf_holder.append(tok)

    page.on("request", capture_csrf)

    print("\nsmartrc に接続中...")
    await page.goto(SMARTRC_URL)
    await page.wait_for_load_state("networkidle", timeout=20000)
    await page.wait_for_timeout(3000)

    csrf = csrf_holder[0] if csrf_holder else ""
    return browser, ctx, page, csrf


# ── メイン ───────────────────────────────────────────────────────

async def main():
    explore_mode = "--explore" in sys.argv
    dump_mode    = "--dump"    in sys.argv

    if explore_mode or dump_mode:
        label      = "調査" if explore_mode else "ダンプ"
        date_input = input(f"{label}対象の開催日を入力してください (例: 20260627): ").strip()
        if len(date_input) != 8 or not date_input.isdigit():
            print("日付の形式が正しくありません。")
            sys.exit(1)

        async with async_playwright() as p:
            browser, ctx, page, csrf = await launch_browser_and_get_csrf(p)
            if not csrf:
                print("⚠ CSRFトークンが取得できませんでした。")
                await browser.close()
                sys.exit(1)

            if explore_mode:
                await explore_fields(ctx, csrf, date_input, page)
            else:
                await dump_all_fields(ctx, csrf, date_input)
            await browser.close()
        return

    # ── 通常取得モード ──────────────────────────────────────────
    print("取得モードを選択してください")
    print("  1: 単日取得")
    print("  2: 期間取得")
    mode = input("モード (1 or 2): ").strip()

    if mode == "1":
        date_input = input("開催日を入力してください (例: 20260419): ").strip()
        if len(date_input) != 8 or not date_input.isdigit():
            print("日付の形式が正しくありません。")
            sys.exit(1)
        target_dates = [date_input]

    elif mode == "2":
        start_input = input("開始日を入力してください (例: 20260101): ").strip()
        end_input   = input("終了日を入力してください (例: 20260419): ").strip()
        if (len(start_input) != 8 or not start_input.isdigit() or
                len(end_input) != 8 or not end_input.isdigit()):
            print("日付の形式が正しくありません。")
            sys.exit(1)
        if start_input > end_input:
            print("開始日が終了日より後になっています。")
            sys.exit(1)
        target_dates = None
    else:
        print("1 か 2 を入力してください。")
        sys.exit(1)

    skip_existing = False
    if mode == "2":
        ans = input("既に取得済みの日付はスキップしますか？ (y/n, デフォルト: y): ").strip().lower()
        skip_existing = (ans != "n")

    async with async_playwright() as p:
        browser, ctx, page, csrf = await launch_browser_and_get_csrf(p)
        if not csrf:
            print("⚠ CSRFトークンが取得できませんでした。")
            await browser.close()
            sys.exit(1)
        print(f"CSRFトークン取得済み: {csrf[:16]}...")

        if mode == "2":
            print(f"\ndays/view で {start_input}〜{end_input} の開催日を検索中...")
            target_dates = await get_race_dates_in_range(ctx, csrf, start_input, end_input)
            if not target_dates:
                print("指定期間内に開催日が見つかりませんでした。")
                await browser.close()
                sys.exit(1)
            print(f"{len(target_dates)} 開催日が見つかりました: {', '.join(target_dates)}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        total   = len(target_dates)
        saved   = 0
        skipped = 0
        failed  = 0

        for i, kaisai_date in enumerate(target_dates, 1):
            out_path = OUT_DIR / f"smartrc_{kaisai_date}.csv"

            print(f"\n[{i}/{total}] {kaisai_date}")

            if skip_existing and out_path.exists():
                print(f"  → 既存ファイルあり（スキップ）: {out_path.name}")
                skipped += 1
                continue

            df = await fetch_one_date(ctx, csrf, kaisai_date)

            if df is None:
                failed += 1
                continue

            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            races  = df[["場所", "R"]].drop_duplicates().shape[0]
            venues = df["場所"].unique().tolist()
            print(f"  → 保存: {out_path.name}  ({len(df)}頭 / {races}R / {', '.join(venues)})")
            saved += 1

            if i < total:
                await asyncio.sleep(1.0)

        await browser.close()

    print(f"\n{'='*40}")
    print(f"完了: 保存 {saved} 日 / スキップ {skipped} 日 / 失敗 {failed} 日")


if __name__ == "__main__":
    asyncio.run(main())
