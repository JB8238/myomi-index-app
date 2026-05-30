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


def get_race_ids_for_date(kaisai_date: str) -> list[str]:
    """
    レース一覧ページから当日の全 race_id リストを取得する。
    /race/shutuba.html?race_id=XXXXXXXXXXXX 形式のリンクを収集。
    """
    url = f"{BASE_URL}/top/race_list.html?kaisai_date={kaisai_date}"
    print(f"レース一覧を取得中: {url}")

    try:
        resp = _SESSION.get(url, timeout=20)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
    except Exception as e:
        print(f"レース一覧ページ取得エラー: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    race_ids: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "shutuba.html" not in href:
            continue
        m = re.search(r"race_id=(\d{12})", href)
        if m:
            rid = m.group(1)
            if rid not in race_ids:
                race_ids.append(rid)

    print(f"  → {len(race_ids)} レースを発見")
    return race_ids


def get_shutuba_info(race_id: str) -> dict | None:
    """
    出馬表ページから馬場状態・騎手情報を取得する。

    馬場状態: .RaceData01 のテキストから「芝：良」「ダ：稍重」パターンを抽出
    騎手名:   /jockey/ へのリンクテキストを馬番と紐付けて取得

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

    try:
        resp = _SESSION.get(url, timeout=20)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
    except Exception as e:
        print(f"  ✗ {venue} {race_r}R: ページ取得エラー: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    result: dict = {
        "race_id": race_id,
        "場所": venue,
        "R": race_r,
        "track_conditions": {},
        "jockeys": {},
    }

    # --- 馬場状態の取得 ---
    # .RaceData01 のテキストに "芝：良" / "ダ：稍重" などが含まれる
    race_data_el = soup.find(class_="RaceData01")
    if race_data_el:
        text = race_data_el.get_text()
        # 全角コロン「：」と半角コロン「:」の両方に対応
        for m in re.finditer(r"(芝|ダ)\s*[：:]\s*(良|稍重|重|不良)", text):
            surface, condition = m.group(1), m.group(2)
            if surface not in result["track_conditions"]:
                result["track_conditions"][surface] = condition

    # --- 騎手名の取得 ---
    # #shutuba_table, .Shutuba_Table, .RaceTable01 のいずれかを試す
    table_body = (
        soup.select_one("#shutuba_table tbody")
        or soup.select_one(".Shutuba_Table tbody")
        or soup.select_one(".RaceTable01 tbody")
    )

    if table_body:
        for tr in table_body.find_all("tr"):
            umaban: int | None = None

            # .Umaban クラス優先
            umaban_el = tr.find(class_="Umaban")
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
