"""
scripts/smartrc_diagnose.py

smartrc.jp のDOM構造とAPIエンドポイントを調査する診断スクリプト。
fetch_smartrc.py の自動化実装に必要な情報を収集する。

Usage:
    python scripts/smartrc_diagnose.py

出力: data/smartrc/diagnose_report.txt
"""

import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright

SMARTRC_URL = "https://www.smartrc.jp/v3/"
OUT_PATH = Path("data/smartrc/diagnose_report.txt")


async def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str = ""):
        print(msg)
        lines.append(msg)

    log("=" * 60)
    log("smartrc 診断レポート")
    log("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        # ── 全JSONレスポンスをキャプチャ ──────────────────────
        json_responses: list[dict] = []

        async def capture_response(response):
            if response.status != 200:
                return
            ct = response.headers.get("content-type", "")
            if "json" not in ct:
                return
            try:
                data = await response.json()
                json_responses.append({
                    "url": response.url,
                    "method": response.request.method,
                    "post_data": response.request.post_data,
                    "data_keys": list(data.keys()) if isinstance(data, dict) else type(data).__name__,
                    "total": data.get("total") if isinstance(data, dict) else None,
                    "data_len": len(data.get("data", [])) if isinstance(data, dict) else None,
                    "success": data.get("success") if isinstance(data, dict) else None,
                    "first_item_keys": (
                        list(data["data"][0].keys())[:15]
                        if isinstance(data, dict) and data.get("data") and isinstance(data["data"], list) and data["data"]
                        else None
                    ),
                })
            except Exception:
                pass

        page.on("response", capture_response)

        # ── ページ遷移 ──────────────────────────────────────────
        log(f"\nURLに移動: {SMARTRC_URL}")
        await page.goto(SMARTRC_URL)
        await page.wait_for_load_state("networkidle", timeout=20000)
        await page.wait_for_timeout(3000)

        log("\n[手動操作] ブラウザで任意のレースを1〜2個クリックしてみてください。")
        log("[手動操作] 終わったら Enter を押してください。")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, input, "  → Enter: ")
        await page.wait_for_timeout(2000)

        # ── キャプチャしたJSONレスポンスを報告 ──────────────────
        log("\n" + "=" * 60)
        log(f"[1] キャプチャしたJSONレスポンス ({len(json_responses)} 件)")
        log("=" * 60)
        for i, r in enumerate(json_responses):
            log(f"\n--- #{i+1} ---")
            log(f"  URL      : {r['url']}")
            log(f"  Method   : {r['method']}")
            log(f"  PostData : {r['post_data']}")
            log(f"  RespKeys : {r['data_keys']}")
            log(f"  success  : {r['success']}")
            log(f"  total    : {r['total']}")
            log(f"  data_len : {r['data_len']}")
            log(f"  item_keys: {r['first_item_keys']}")

        # ── クリッカブル要素の調査 ────────────────────────────────
        log("\n" + "=" * 60)
        log("[2] クリッカブル要素（ボタン・タブ・リンク）")
        log("=" * 60)

        elements = await page.evaluate("""() => {
            const sel = 'button, a, [role="tab"], [role="button"], ' +
                        '[class*="tab"], [class*="race"], [class*="btn"]';
            return Array.from(document.querySelectorAll(sel))
                .filter(el => {
                    const txt = el.textContent.trim();
                    const rect = el.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0 && txt.length > 0;
                })
                .slice(0, 80)
                .map(el => ({
                    tag:   el.tagName,
                    text:  el.textContent.trim().substring(0, 40),
                    cls:   el.className.substring(0, 80),
                    id:    el.id,
                    role:  el.getAttribute('role'),
                    href:  el.href || null,
                }));
        }""")

        for el in elements:
            log(f"  [{el['tag']}] text={repr(el['text'])} cls={repr(el['cls'])} id={repr(el['id'])} role={repr(el['role'])}")

        # ── タブ系要素に絞って追加調査 ────────────────────────────
        log("\n" + "=" * 60)
        log("[3] 'R' を含むテキストを持つ要素")
        log("=" * 60)

        r_elements = await page.evaluate("""() => {
            const all = document.querySelectorAll('*');
            const results = [];
            for (const el of all) {
                if (results.length >= 50) break;
                const direct = Array.from(el.childNodes)
                    .filter(n => n.nodeType === 3)
                    .map(n => n.textContent.trim())
                    .join('');
                if (/^\\d{1,2}R$/.test(direct) || /^R\\d{1,2}$/.test(direct)) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        results.push({
                            tag:  el.tagName,
                            text: direct,
                            cls:  el.className.substring(0, 80),
                            id:   el.id,
                            role: el.getAttribute('role'),
                        });
                    }
                }
            }
            return results;
        }""")

        if r_elements:
            log(f"  {len(r_elements)} 件見つかりました:")
            for el in r_elements:
                log(f"  [{el['tag']}] text={repr(el['text'])} cls={repr(el['cls'])} id={repr(el['id'])} role={repr(el['role'])}")
        else:
            log("  'nR' 形式のテキストを持つ可視要素は見つかりませんでした。")

        # ── ExtJS/グローバル変数の調査 ────────────────────────────
        log("\n" + "=" * 60)
        log("[4] グローバルJS変数 (Ext / SG / App)")
        log("=" * 60)

        globals_info = await page.evaluate("""() => {
            const info = {};
            info.hasExt = typeof Ext !== 'undefined';
            info.hasSG  = typeof SG  !== 'undefined';
            info.hasApp = typeof App !== 'undefined';
            if (typeof Ext !== 'undefined') {
                try {
                    const tabs = Ext.ComponentQuery.query('tab');
                    info.extTabs = tabs.slice(0, 20).map(t => ({
                        text: t.getText ? t.getText() : t.text,
                        id: t.id,
                        cls: t.cls,
                    }));
                } catch(e) { info.extTabsError = String(e); }
                try {
                    const panels = Ext.ComponentQuery.query('tabpanel');
                    info.extTabPanels = panels.length;
                } catch(e) {}
            }
            return info;
        }""")
        log(f"  Ext.js  : {globals_info.get('hasExt')}")
        log(f"  SG      : {globals_info.get('hasSG')}")
        log(f"  App     : {globals_info.get('hasApp')}")
        if globals_info.get("extTabs"):
            log(f"  ExtJS tabs ({len(globals_info['extTabs'])} 件):")
            for t in globals_info["extTabs"]:
                log(f"    text={repr(t['text'])} id={repr(t['id'])} cls={repr(t['cls'])}")
        if globals_info.get("extTabsError"):
            log(f"  ExtJS tab query error: {globals_info['extTabsError']}")
        log(f"  TabPanels: {globals_info.get('extTabPanels', 'N/A')}")

        await browser.close()

    # ── レポート保存 ───────────────────────────────────────────
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n✅ レポート保存: {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
