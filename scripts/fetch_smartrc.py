"""
scripts/fetch_smartrc.py

smartrc.jp から「推定人気」「人気ランク」を取得する。
単日取得と期間取得の両方に対応。

Usage:
    python scripts/fetch_smartrc.py

出力: data/smartrc/smartrc_YYYYMMDD.csv
  列: 開催日, 場所, R, 馬番, 馬名, 推定人気, 人気ランク
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


# ── 1開催日分のデータ取得 ────────────────────────────────────────

async def fetch_one_date(ctx, csrf: str, kaisai_date: str) -> pd.DataFrame | None:
    """1開催日分の全レース・全馬データを取得して DataFrame を返す。"""
    venue_codes = get_venue_codes(kaisai_date)

    # races/view で各会場のrcode一覧を取得
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

    # runners/view で各レースの馬データを取得
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
            all_rows.append({
                "開催日":     int(date_str),
                "場所":       venue_name,
                "R":          race_num,
                "馬番":       int(horse["uno"]),
                "馬名":       horse.get("hname", ""),
                "推定人気":   int(horse["est_pop"]) if horse.get("est_pop") else None,
                "人気ランク": horse.get("old_pr"),
            })
        print(f"    ✓ 取得: {venue_name} {race_num}R ({len(data['data'])}頭)")

        await asyncio.sleep(0.3)  # API負荷軽減

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["開催日", "場所", "R", "馬番"])
    df = df.sort_values(["場所", "R", "馬番"]).reset_index(drop=True)
    return df


# ── days/view から期間内の開催日リストを取得 ───────────────────────

async def get_race_dates_in_range(ctx, csrf: str, start_date: str, end_date: str) -> list[str]:
    """days/view API で期間内の実際の開催日リストを返す。"""
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


# ── メイン ───────────────────────────────────────────────────────

async def main():
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
        target_dates = None  # days/view で後から取得
    else:
        print("1 か 2 を入力してください。")
        sys.exit(1)

    # 既存ファイルのスキップ確認（期間取得のみ）
    skip_existing = False
    if mode == "2":
        ans = input("既に取得済みの日付はスキップしますか？ (y/n, デフォルト: y): ").strip().lower()
        skip_existing = (ans != "n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        ctx     = await browser.new_context()
        page    = await ctx.new_page()

        # CSRF トークンをリクエストヘッダから取得
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
        if not csrf:
            print("⚠ CSRFトークンが取得できませんでした。")
            await browser.close()
            sys.exit(1)
        print(f"CSRFトークン取得済み: {csrf[:16]}...")

        # 期間取得: days/view で開催日リストを確定
        if mode == "2":
            print(f"\ndays/view で {start_input}〜{end_input} の開催日を検索中...")
            target_dates = await get_race_dates_in_range(ctx, csrf, start_input, end_input)
            if not target_dates:
                print("指定期間内に開催日が見つかりませんでした。")
                await browser.close()
                sys.exit(1)
            print(f"{len(target_dates)} 開催日が見つかりました: {', '.join(target_dates)}")

        # 各開催日のデータ取得
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        total    = len(target_dates)
        saved    = 0
        skipped  = 0
        failed   = 0

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
                await asyncio.sleep(1.0)  # 日付間のインターバル

        await browser.close()

    print(f"\n{'='*40}")
    print(f"完了: 保存 {saved} 日 / スキップ {skipped} 日 / 失敗 {failed} 日")


if __name__ == "__main__":
    asyncio.run(main())
