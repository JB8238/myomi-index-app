"""
scripts/fetch_netkeiba_changes.py

netkeiba.com から当日の馬場状態・騎手情報を取得し、
preprocessed_data_YYYYMMDD.csv との差分（変更点）を検出して JSON に保存する。

requests + BeautifulSoup によるブラウザ不要の実装。

Usage:
    python scripts/fetch_netkeiba_changes.py YYYYMMDD

Output:
    data/tmp/netkeiba_changes_YYYYMMDD.json

Notes:
    - 馬場状態の変更検出対象: 稍重→良 / 良→稍重 のみ
    - 騎手変更は (場所, R, 馬番) をキーに全変更を検出
    - netkeiba の HTML 構造が変わった場合は get_shutuba_info() のセレクタを調整すること
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

VENUE_CODE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

TMP_DIR = Path("data/tmp")
BASE_URL = "https://race.netkeiba.com"

# ブラウザに見せかけたヘッダー
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Referer": "https://race.netkeiba.com/",
}

_SESSION = requests.Session()
_SESSION.headers.update(_HEADERS)


def decode_race_id(race_id: str) -> dict | None:
    """race_id（12桁）から場所・レース番号を解読する。"""
    if len(race_id) != 12 or not race_id.isdigit():
        return None
    venue_code = race_id[4:6]
    race_num = int(race_id[10:12])
    venue_name = VENUE_CODE_MAP.get(venue_code)
    if venue_name is None:
        return None
    return {"場所": venue_name, "R": race_num}


def normalize_condition(val: str) -> str:
    """馬場状態から (暫定)/(確定) を除去して正規化する。"""
    if not isinstance(val, str):
        return ""
    return val.replace("(暫定)", "").replace("(確定)", "").strip()


def _fetch_html(url: str) -> str | None:
    """URL を取得して EUC-JP / UTF-8 を自動判定しながらデコードして返す。"""
    try:
        resp = _SESSION.get(url, timeout=20)
        resp.raise_for_status()
        for enc in (resp.apparent_encoding, "euc-jp", "utf-8", "shift-jis"):
            if not enc:
                continue
            try:
                return resp.content.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return resp.text
    except Exception as e:
        print(f"  取得エラー ({url}): {e}")
        return None


def _extract_race_ids(html: str, year: str) -> list[str]:
    """HTML テキストから year で始まる 12 桁の race_id を抽出する。"""
    race_ids: list[str] = []
    # パターン 1: race_id=XXXXXXXXXXXX (クエリパラメータ形式)
    for m in re.finditer(r"race_id[=%%\"']+(\d{12})", html):
        rid = m.group(1)
        if rid.startswith(year) and rid not in race_ids:
            race_ids.append(rid)
    # パターン 2: 12桁数値がそのまま埋め込まれている (JavaScript / JSON / data 属性)
    if not race_ids:
        for m in re.finditer(rf"\b({year}(?:0[1-9]|10)\d{{6}})\b", html):
            rid = m.group(1)
            if rid not in race_ids:
                race_ids.append(rid)
    return race_ids


def get_race_ids_for_date(kaisai_date: str) -> list[str]:
    """
    当日の全 race_id リストを取得する。

    試行順:
      1. race_list.html の HTML 全体から抽出
      2. race_list_sub.html (部分HTML/Ajaxコンテンツ) から抽出
      3. api_get_race_list.html (JSON API) から抽出
    """
    year = kaisai_date[:4]
    race_ids: list[str] = []

    # --- 試行 1: メインのレース一覧ページ ---
    url1 = f"{BASE_URL}/top/race_list.html?kaisai_date={kaisai_date}"
    print(f"レース一覧を取得中: {url1}")
    html1 = _fetch_html(url1)
    if html1:
        print(f"  レスポンス: {len(html1)} 文字")
        race_ids = _extract_race_ids(html1, year)

    # --- 試行 2: サブHTML (Ajax コンテンツ) ---
    if not race_ids:
        url2 = f"{BASE_URL}/top/race_list_sub.html?kaisai_date={kaisai_date}"
        print(f"  → サブHTMLを試みます: {url2}")
        html2 = _fetch_html(url2)
        if html2:
            print(f"    レスポンス: {len(html2)} 文字")
            race_ids = _extract_race_ids(html2, year)

    # --- 試行 3: JSON API エンドポイント ---
    if not race_ids:
        url3 = f"{BASE_URL}/api/api_get_race_list.html?kaisai_date={kaisai_date}"
        print(f"  → APIを試みます: {url3}")
        html3 = _fetch_html(url3)
        if html3:
            print(f"    レスポンス: {len(html3)} 文字")
            race_ids = _extract_race_ids(html3, year)

    # --- デバッグ: 全試行失敗時に HTML 本文部分を出力 ---
    if not race_ids and html1:
        # 先頭の HEAD を読み飛ばして BODY 付近 (5000文字目から) を表示
        mid_start = min(5000, len(html1) // 3)
        snippet = html1[mid_start:mid_start + 1200].replace("\n", " ").replace("\r", "")
        print(f"  [DEBUG] HTML 本文付近 ({mid_start} 文字目から 1200 文字):")
        for i in range(0, len(snippet), 120):
            print(f"    {snippet[i:i+120]}")

    print(f"  → {len(race_ids)} レースを発見")
    return race_ids


def get_shutuba_info(race_id: str) -> dict | None:
    """
    出馬表ページから馬場状態・騎手情報を取得する。

    馬場状態: .RaceData01 のテキストから以下の2パターンを組み合わせて抽出
        - 馬場面: "ダ1400m" / "芝2000m" → 芝 or ダ
        - 馬場状態: "馬場:良" / "馬場：稍重" → 良|稍重|重|不良
    騎手名:   /jockey/ へのリンクテキストを馬番と紐付けて取得
              ※ html.parser は <tbody> を自動挿入しないため tbody なしでテーブルを検索

    Returns:
        {
            "race_id": str,
            "場所": str,
            "R": int,
            "track_conditions": {"芝": str, "ダ": str},   # 取得できたもののみ
            "jockeys": {馬番(int): 騎手名(str)}
        }
    """
    decoded = decode_race_id(race_id)
    if not decoded:
        return None

    venue, race_r = decoded["場所"], decoded["R"]
    url = f"{BASE_URL}/race/shutuba.html?race_id={race_id}"

    html = _fetch_html(url)
    if html is None:
        print(f"  ✗ {venue} {race_r}R: ページ取得失敗")
        return None

    soup = BeautifulSoup(html, "html.parser")

    result: dict = {
        "race_id": race_id,
        "場所": venue,
        "R": race_r,
        "track_conditions": {},
        "jockeys": {},
    }

    # --- 馬場状態の取得 ---
    # .RaceData01 テキスト例:
    #   "09:50発走 / ダ1400m (左)\n/ 天候:晴\n/ 馬場:良"
    #   "10:05発走 / 芝2000m\n/ 天候:曇\n/ 馬場:稍重"
    race_data_el = soup.find(class_="RaceData01")
    if race_data_el:
        text = race_data_el.get_text()
        # 馬場状態: "馬場:良" / "馬場：稍重"
        cond_m = re.search(r"馬場[：:]\s*(良|稍重|重|不良)", text)
        # 馬場面: "ダXXXXm" or "芝XXXXm"
        surf_m = re.search(r"(芝|ダ)\d+m", text)
        if cond_m and surf_m:
            result["track_conditions"][surf_m.group(1)] = cond_m.group(1)

    # --- 騎手名の取得 ---
    # html.parser は <tbody> を暗黙挿入しないため tbody なしでテーブルを取得する
    table = (
        soup.select_one(".Shutuba_Table")
        or soup.select_one(".ShutubaTable")
        or soup.select_one(".RaceTable01")
    )

    if table:
        for tr in table.find_all("tr"):
            umaban: int | None = None

            # .Umaban クラス優先（Umaban1, Umaban2 等の番号付きも対応）
            umaban_el = tr.find(class_="Umaban") or tr.find(
                class_=re.compile(r"^Umaban\d+$")
            )
            if umaban_el:
                try:
                    umaban = int(umaban_el.get_text().strip())
                except ValueError:
                    pass

            # フォールバック: 2列目のテキスト
            if umaban is None:
                cells = tr.find_all("td")
                if len(cells) >= 2:
                    try:
                        umaban = int(cells[1].get_text().strip())
                    except ValueError:
                        pass

            if umaban is None:
                continue

            # /jockey/ URL のリンクテキストを騎手名として使用
            # db.netkeiba.com/jockey/result/recent/XXXXX/ 形式にも対応
            jockey_link = tr.find("a", href=re.compile(r"/jockey/"))
            if jockey_link:
                jockey_name = jockey_link.get_text().strip()
            else:
                # フォールバック: .Jockey クラスのセル
                jockey_cell = tr.find(class_="Jockey")
                jockey_name = jockey_cell.get_text().strip() if jockey_cell else ""

            if jockey_name:
                result["jockeys"][umaban] = jockey_name

    cond_str = ", ".join(f"{s}:{c}" for s, c in result["track_conditions"].items()) or "取得なし"
    print(f"  ✓ {venue} {race_r}R: 馬場=[{cond_str}], 騎手={len(result['jockeys'])}頭")
    return result


def detect_changes(kaisai_date: str, race_infos: list[dict]) -> dict:
    """
    スクレイプ結果と既存 preprocessed_data CSV を比較して変更を検出する。

    馬場状態変更: 良↔稍重 の遷移のみ対象（(場所, 種別) 単位で集計）
    騎手変更: (場所, R, 馬番) ごとに検出

    Returns: changes dict（JSON 出力用）
    """
    year = kaisai_date[:4]
    csv_path = Path(f"data/{year}/{kaisai_date}/preprocessed_data_{kaisai_date}.csv")
    if not csv_path.exists():
        return {
            "error": f"CSVファイルが見つかりません: {csv_path}",
            "kaisai_date": kaisai_date,
            "track_changes": [],
            "jockey_changes": [],
        }

    df = pd.read_csv(csv_path, encoding="utf-8")

    # スクレイプ結果を統合 -----------------------------------------------
    # 馬場状態: {場所: {"芝": "良", "ダ": "稍重", ...}}
    venue_track: dict[str, dict[str, str]] = {}
    # 騎手: {(場所, R, 馬番): 騎手名}
    jockey_map: dict[tuple, str] = {}

    for info in race_infos:
        venue = info["場所"]
        race_r = info["R"]
        if venue not in venue_track:
            venue_track[venue] = {}
        for surface, cond in info["track_conditions"].items():
            # 同一 venue の同一面は最初の取得値を優先
            if surface not in venue_track[venue]:
                venue_track[venue][surface] = cond
        for umaban, jockey in info["jockeys"].items():
            jockey_map[(venue, race_r, int(umaban))] = jockey

    # 馬場状態の変更検出 -------------------------------------------------
    track_changes: list[dict] = []

    for (venue, surface_type), group in df.groupby(["場所", "種別"]):
        # 種別 → netkeiba の馬場面キーに変換
        netkeiba_key = "芝" if surface_type == "芝" else "ダ"

        if venue not in venue_track or netkeiba_key not in venue_track[venue]:
            continue

        current_cond_raw = str(group["馬場状態"].iloc[0])
        current_cond_norm = normalize_condition(current_cond_raw)
        new_cond = venue_track[venue][netkeiba_key]
        new_cond_norm = normalize_condition(new_cond)

        # 良↔稍重 の変化のみ対象
        if (current_cond_norm == "良" and new_cond_norm == "稍重") or (
            current_cond_norm == "稍重" and new_cond_norm == "良"
        ):
            track_changes.append(
                {
                    "場所": venue,
                    "種別": surface_type,
                    "旧馬場状態": current_cond_raw,
                    "新馬場状態": new_cond,
                    "影響行数": len(group),
                }
            )

    # 騎手変更の検出 -----------------------------------------------------
    jockey_changes: list[dict] = []

    for _, row in df.iterrows():
        venue = row["場所"]
        race_r = int(row["R"])
        umaban = int(row["馬番"])
        key = (venue, race_r, umaban)
        if key not in jockey_map:
            continue
        new_jockey = jockey_map[key].strip()
        current_jockey = str(row.get("騎手名", "")).strip()
        if new_jockey and new_jockey != current_jockey:
            jockey_changes.append(
                {
                    "場所": venue,
                    "R": race_r,
                    "馬番": umaban,
                    "馬名": row.get("馬名", ""),
                    "旧騎手名": current_jockey,
                    "新騎手名": new_jockey,
                }
            )

    return {
        "kaisai_date": kaisai_date,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "track_changes": track_changes,
        "jockey_changes": jockey_changes,
    }


def main(kaisai_date: str) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    race_ids = get_race_ids_for_date(kaisai_date)
    if not race_ids:
        print("レースが見つかりませんでした。")
        return

    # preprocessed CSV が存在する場合、対象会場に絞る
    year = kaisai_date[:4]
    csv_path = Path(f"data/{year}/{kaisai_date}/preprocessed_data_{kaisai_date}.csv")
    target_venues: set[str] = set()
    if csv_path.exists():
        df_prep = pd.read_csv(csv_path, encoding="utf-8")
        target_venues = set(df_prep["場所"].unique())
        race_ids = [
            rid for rid in race_ids
            if (d := decode_race_id(rid)) and d["場所"] in target_venues
        ]
        print(f"対象会場: {', '.join(sorted(target_venues))}")
        print(f"対象レース数: {len(race_ids)}")

    # 各レースの出馬表を取得
    race_infos: list[dict] = []
    for i, race_id in enumerate(race_ids, 1):
        decoded = decode_race_id(race_id)
        label = f"{decoded['場所']} {decoded['R']}R" if decoded else race_id
        print(f"[{i}/{len(race_ids)}] {label} (race_id={race_id})")
        info = get_shutuba_info(race_id)
        if info:
            race_infos.append(info)
        time.sleep(0.5)  # サーバー負荷軽減

    print("\n変更を検出中...")
    changes = detect_changes(kaisai_date, race_infos)

    out_path = TMP_DIR / f"netkeiba_changes_{kaisai_date}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    print(f"\n結果を保存しました: {out_path}")
    print(f"  馬場状態変更: {len(changes.get('track_changes', []))} 件")
    print(f"  騎手変更:     {len(changes.get('jockey_changes', []))} 件")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python scripts/fetch_netkeiba_changes.py YYYYMMDD")
        sys.exit(1)
    _date = sys.argv[1].strip()
    if len(_date) != 8 or not _date.isdigit():
        print("日付の形式が正しくありません。例: 20260523")
        sys.exit(1)
    main(_date)
