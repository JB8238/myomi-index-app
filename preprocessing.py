import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# 標準出力がファイル/パイプにリダイレクトされるとcp932等にフォールバックし、
# 絵文字混じりのprint()がUnicodeEncodeErrorで落ちうるため明示的にUTF-8化する。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.horse_history import load_horse_race_history, build_zensou_features
from core.ck_features import load_ck_data, attach_ck_features, CK_KEY_FIELDS

JVLINK_DATA_OUT = Path(__file__).resolve().parent.parent / "妙味度指数_jvlink" / "data_out"

PLACE_CODE_TO_NAME = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

RACE_KEY = ["開催年", "開催月日", "競馬場コード", "開催回", "開催日目", "レース番号"]
VENUE_KEY = ["開催年", "開催月日", "競馬場コード", "開催回", "開催日目"]  # WE(天候馬場状態)は開催場単位

GRADE_STAKES_CODES = {"A", "B", "C"}  # A=G1, B=G2, C=G3（実データで確認済み）
BABA_JOTAI_MAP = {"1": "良", "2": "稍重", "3": "重", "4": "不良"}


def _classify_class(grade_code: str, condition_code: str) -> str:
    """グレードコード・競走条件コード(最若年)からクラスを判定する（実データで検証済み）"""
    if grade_code in GRADE_STAKES_CODES:
        return "重賞"
    if condition_code == "701":
        return "新馬"
    if condition_code == "703":
        return "未勝利"
    if condition_code in ("005", "010", "016"):
        return "自己条件"
    return "OP"


def _classify_track_type(track_code) -> str | None:
    try:
        code = int(track_code)
    except (ValueError, TypeError):
        return None
    if 10 <= code <= 22:
        return "芝"
    if 23 <= code <= 29:
        return "ダート"
    if 51 <= code <= 59:
        return "障害"
    return None


def _classify_baba_jotai(
    track_type, ra_turf_code, ra_dirt_code, we_turf_code=None, we_dirt_code=None,
    allow_we_fallback: bool = True,
) -> str | None:
    """
    種別（芝/ダート/障害）に対応するRA(蓄積系)の馬場状態コードを優先し、未確定("0")の
    場合はWE(速報系、JVRTOpen dataspec="0B14"経由)の値で補う。レース確定前（当日の
    出馬表段階）はRA側が常に"0"のため、当日はWE側が実質的な情報源になる。

    ★WEは開催場単位で芝・ダート両方の状態が同時に非ゼロで返るため、RAのように
    「非ゼロの方を採用」では種別を無視してしまう（ダートレースなのに芝の状態を誤採用等）。
    必ず種別で芝/ダートのどちらのコードを見るか決める（障害は芝の状態を代用する）。

    allow_we_fallback=False の場合、RA未確定("0")でもWEにはフォールバックせずNoneを返す。
    発走時刻を過ぎているのにRAが未確定＝そのレースが終わった後にra.csvを再取得して
    いない（古いまま）ことを意味し、その状態でWEの「現在の」値を使うと、当日中に
    馬場状態が変化した場合に既に終わったレースの値まで最新の速報値で一律上書き
    されてしまう（2026-08-02の運用で発覚した不具合）。呼び出し側が発走時刻を見て
    already_started（=発走済みなのにRAが未確定）の行だけFalseを渡すことでこれを防ぐ。
    """
    is_dirt = track_type == "ダート"
    ra_code = ra_dirt_code if is_dirt else ra_turf_code
    if ra_code not in (None, "", "0"):
        return BABA_JOTAI_MAP.get(ra_code)
    if not allow_we_fallback:
        return None
    we_code = we_dirt_code if is_dirt else we_turf_code
    return BABA_JOTAI_MAP.get(we_code)


def _post_time_passed(kaisai_year, kaisai_month_day, hassou_jikoku, now: datetime | None = None) -> bool:
    """開催年+開催月日+発走時刻(HHMM)からレース発走時刻を組み立て、現在時刻を過ぎていればTrue。
    発走時刻が未確定・パース不能な場合はFalse（＝安全側。WEフォールバックを塞がない
    デフォルトに倒す。発走時刻はRAに通常入っているため、この分岐に落ちるのは稀）。
    """
    now = now or datetime.now()
    hm = str(hassou_jikoku).strip() if hassou_jikoku is not None else ""
    if not hm or not hm.isdigit():
        return False  # 発走時刻が未確定・空欄ならパースできたことにしない（安全側）
    try:
        y = int(kaisai_year)
        md = str(kaisai_month_day).strip().zfill(4)
        hm = hm.zfill(4)
        race_dt = datetime(y, int(md[:2]), int(md[2:]), int(hm[:2]), int(hm[2:]))
    except (ValueError, TypeError):
        return False
    return now >= race_dt


def main():

    kaisai_date = input("開催日を入力してください (例: 20240601): ")
    data_pattern = int(input("データパターンを選択してください（1: 確定後データ, 2: 確定前データ）: "))
    year = kaisai_date[:4]
    month_day = kaisai_date[4:]

    DATA_DIR = Path(".", "data", year, kaisai_date)
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    BASE_DATA_DIR = Path("C:/TFJV/TXT/data/base_data", year)
    PEDS_DATA_DIR = Path("C:/TFJV/TXT/data/peds_data", year)

    # -----------------------------
    # JV-Link由来（当日のレース・出走馬情報）
    # -----------------------------

    ra_df = pd.read_csv(JVLINK_DATA_OUT / "ra.csv", encoding="utf-8-sig", dtype=str)
    se_df = pd.read_csv(JVLINK_DATA_OUT / "se.csv", encoding="utf-8-sig", dtype=str)

    ra_df = ra_df[(ra_df["開催年"] == year) & (ra_df["開催月日"] == month_day)].copy()
    se_df = se_df[(se_df["開催年"] == year) & (se_df["開催月日"] == month_day)].copy()

    if ra_df.empty or se_df.empty:
        raise FileNotFoundError(
            f"{JVLINK_DATA_OUT} にこの開催日のRA/SEが見つかりません。"
            f"先に妙味度指数_jvlink側で fetch_race_data.py を実行してください。"
        )

    ra_small = ra_df[RACE_KEY + [
        "距離", "グレードコード", "競走条件コード_最若年", "トラックコード",
        "芝馬場状態コード", "ダート馬場状態コード", "発走時刻",
    ]].drop_duplicates(subset=RACE_KEY)

    entries = se_df.merge(ra_small, on=RACE_KEY, how="inner")
    entries = entries[entries["競馬場コード"].isin(PLACE_CODE_TO_NAME)].copy()

    entries["場所"] = entries["競馬場コード"].map(PLACE_CODE_TO_NAME)
    entries["R"] = pd.to_numeric(entries["レース番号"], errors="coerce")
    entries["距離"] = pd.to_numeric(entries["距離"], errors="coerce")
    entries["馬番"] = pd.to_numeric(entries["馬番"], errors="coerce")

    entries["クラス"] = entries.apply(
        lambda r: _classify_class(r["グレードコード"], r["競走条件コード_最若年"]), axis=1
    )
    entries["種別"] = entries["トラックコード"].apply(_classify_track_type)

    # 天候馬場状態（速報系WE、JVRTOpen dataspec="0B14"経由）: レース確定前はRAの馬場状態
    # コードが未確定("0")のため、当日はこちらが実質的な情報源になる。無ければRAのみで判定。
    we_path = JVLINK_DATA_OUT / "we.csv"
    if we_path.exists():
        we_df = pd.read_csv(we_path, encoding="utf-8-sig", dtype=str)
        we_df = we_df[(we_df["開催年"] == year) & (we_df["開催月日"] == month_day)].copy()
        we_small = we_df[VENUE_KEY + ["馬場状態・芝", "馬場状態・ダート"]].rename(
            columns={"馬場状態・芝": "WE芝馬場状態コード", "馬場状態・ダート": "WEダート馬場状態コード"}
        )
        entries = entries.merge(we_small, on=VENUE_KEY, how="left")
    else:
        entries["WE芝馬場状態コード"] = None
        entries["WEダート馬場状態コード"] = None

    entries["発走済み"] = entries.apply(
        lambda r: _post_time_passed(r["開催年"], r["開催月日"], r.get("発走時刻")), axis=1
    )

    entries["馬場状態"] = entries.apply(
        lambda r: _classify_baba_jotai(
            r["種別"], r["芝馬場状態コード"], r["ダート馬場状態コード"],
            r["WE芝馬場状態コード"], r["WEダート馬場状態コード"],
            allow_we_fallback=not r["発走済み"],
        ), axis=1
    )

    # 発走済みなのにRAが未確定("0")のまま＝ra.csvがそのレースの終了後に再取得されて
    # いない状態。WEの現在値で上書きせず空にしたので、再取得を促す警告を出す。
    # ★2026-08-26判明: --option 2（今週データ・増分配信）は同日中に再実行しても、
    #   既に一度配信済みのレースの確定ステータス更新（未確定"0"→確定コード）を
    #   再送してくれないことがある（エラーにはならず、静かに古いままになる）。
    #   fetch_race_data.py側は既にアップサート方式（既存データを消さず統合）なので、
    #   --option 4（セットアップデータ・常に現在の全件を返す）を使えば安全かつ確実に
    #   最新の確定ステータスを取得できる。
    _stale = entries[entries["発走済み"] & entries["馬場状態"].isna()]
    if not _stale.empty:
        _races = sorted(set(zip(_stale["場所"], _stale["R"].astype("Int64"))))
        print(
            f"⚠ 発走時刻を過ぎているのにRAの馬場状態が未確定のレースがあります（ra.csvが古い可能性）: {_races}\n"
            f"  馬場状態は空のままにしました（WEの現在値で誤って上書きしないため）。"
            f"  妙味度指数_jvlink側で fetch_race_data.py --sid UNKNOWN --from <本日>000000 --option 4 を再実行し、"
            f"  ra.csv/se.csvを最新化してからpreprocessing.pyをやり直してください"
            f"（--option 2は確定ステータスの更新を再送しないことがあるため、再取得時は--option 4を推奨）。"
        )

    l = []
    for age in entries["馬齢"]:
        try:
            a = int(age)
        except (ValueError, TypeError):
            l.append(np.nan)
            continue
        l.append(f"{a}歳" if a <= 4 else "5歳以上")
    entries["年齢"] = l

    entries = entries.rename(columns={"騎手名略称": "騎手名", "調教師名略称": "調教師名"})

    # 騎手変更（速報系JC、WEと同じdataspec="0B14"経由）: SEの騎手名略称は再取得するまで
    # 変更前の値のままになるため、JCで捕捉された変更後の騎手名があれば優先する。JCは
    # レース・馬番単位の確定した変更通知であり、WEの馬場状態のように「今この瞬間の値」が
    # 時間とともに変わる性質ではないため、発走時刻チェック（allow_we_fallback相当）は不要。
    jc_path = JVLINK_DATA_OUT / "jc.csv"
    if jc_path.exists():
        jc_df = pd.read_csv(jc_path, encoding="utf-8-sig", dtype=str)
        jc_df = jc_df[(jc_df["開催年"] == year) & (jc_df["開催月日"] == month_day)].copy()
        jc_small = jc_df[RACE_KEY + ["馬番", "騎手名"]].rename(columns={"騎手名": "JC騎手名"})
        jc_small["馬番"] = pd.to_numeric(jc_small["馬番"], errors="coerce")
        entries = entries.merge(jc_small, on=RACE_KEY + ["馬番"], how="left")
        entries["騎手名"] = entries["JC騎手名"].combine_first(entries["騎手名"])
        entries = entries.drop(columns=["JC騎手名"])

    # -----------------------------
    # JV-Link由来（出走別着度数CK: 脚質傾向・コース/距離適性）
    # -----------------------------
    ck_path = JVLINK_DATA_OUT / "ck.csv"
    if ck_path.exists():
        ck_df = load_ck_data(ck_path)
        ck_df = ck_df[(ck_df["開催年"] == year) & (ck_df["開催月日"] == month_day)].copy()
        entries = attach_ck_features(entries, ck_df)
    else:
        entries["脚質傾向"] = np.nan
        entries["コース複勝率"] = np.nan
        entries["距離帯複勝率"] = np.nan

    base_pre = entries[[
        "場所", "R", "クラス", "種別", "距離", "馬場状態",
        "馬番", "馬名", "年齢", "騎手名", "調教師名",
        "脚質傾向", "コース複勝率", "距離帯複勝率",
    ]].reset_index(drop=True)

    # 距離区分
    l = []
    for i in range(len(base_pre)):
        if base_pre.loc[i, "距離"] <= 1600:
            l.append("短距離")
        elif base_pre.loc[i, "距離"] <= 2200:
            l.append("中距離")
        else:
            l.append("長距離")
    base_pre["距離区分"] = l

    # 回り
    l = []
    left_tern = re.compile("東京|新潟|中京")
    for i in range(len(base_pre)):
        if left_tern.search(base_pre.loc[i, "場所"]):
            l.append("左回り")
        else:
            l.append("右回り")
    base_pre["回り"] = l

    # 道悪判定
    # ★障害は_classify_baba_jotaiの時点で芝の馬場状態を代用している（マスタに障害専用の
    #   道悪列が無いため）ので、ラベルも芝道悪にする必要がある。以前は「芝以外は全部ダ道悪」
    #   というelse節になっており、障害レースが実際には芝の値を見ているのに「ダ道悪」の
    #   ラベルが付き、騎手・調教師・種牡馬の道悪判定利益度もダートの列で誤って照合していた
    #   （2026-08-26発覚、693行該当）。
    l = []
    for i in range(len(base_pre)):
        baba = base_pre.loc[i, "馬場状態"]
        if pd.isna(baba) or baba == "良":
            l.append(np.nan)
        elif base_pre.loc[i, "種別"] == "ダート":
            l.append("ダ道悪")
        else:
            l.append("芝道悪")
    base_pre["道悪判定"] = l

    # -----------------------------
    # レースレベル（TARGET手動出力を継続利用。JV-Dataに相当する値が無いため）
    # -----------------------------

    base_df = pd.read_csv(
        BASE_DATA_DIR / f"base_data_{kaisai_date}.csv",
        encoding="shift-jis",
        header=None,
    )
    # 場所(3),R(4),馬番(14),レースレベル(11)は日付・データパターンによらず共通の列位置
    level_pre = base_df.iloc[:, [3, 4, 14, 11]].copy()
    level_pre.columns = ["場所", "R", "馬番", "レースレベル"]
    level_pre["R"] = pd.to_numeric(level_pre["R"], errors="coerce")
    level_pre["馬番"] = pd.to_numeric(level_pre["馬番"], errors="coerce")

    base_df_preprocessed = base_pre.merge(level_pre, on=["場所", "R", "馬番"], how="left")

    # -----------------------------
    # 血統（TARGET手動出力を継続利用。UMマスタ対応はPhase 2）
    # -----------------------------

    peds_df = pd.read_csv(
        PEDS_DATA_DIR / f"peds_data_{kaisai_date}.csv",
        encoding="shift-jis",
        header=None,
    )
    if data_pattern == 1:
        # 2026-08-26に発覚: 2025-11-10以降もpeds_dataの列数は34列のまま変わって
        # いないが、「馬名」の位置だけ19→21に戻る一方で「種牡馬名(父)」の位置は
        # 30のまま変わっていない（sire_prof_listとの突き合わせで実データを確認済み:
        # col30の一致率91-99%=父、col32の一致率46-51%=母父）。
        # 旧コードは2025-11-09以前の36列時代の位置(21,32)にそのまま戻していたため、
        # 2025-11-10以降のdata_pattern=1では実際には母父を種牡馬名として読んでいた。
        if int(kaisai_date) < 20250215:
            peds_df_pre = peds_df.iloc[:, [21, 32]]
        elif int(kaisai_date) <= 20251109:
            peds_df_pre = peds_df.iloc[:, [19, 30]]
        else:
            peds_df_pre = peds_df.iloc[:, [21, 30]]
    elif data_pattern == 2:
        if int(kaisai_date) >= 20250215 and int(kaisai_date) <= 20251109:
            peds_df_pre = peds_df.iloc[:, [19, 28]]
        else:
            peds_df_pre = peds_df.iloc[:, [21, 30]]
    else:
        raise ValueError("不正な値が入力されています。'1'か'2'を入力してください。")

    peds_df_pre = peds_df_pre.copy()
    peds_df_pre.columns = ["馬名", "種牡馬名"]

    base_df_preprocessed = base_df_preprocessed.merge(peds_df_pre, on="馬名", how="left")

    # -----------------------------
    # 前走データ（JV-Link馬別レース履歴から算出）
    # -----------------------------

    history_df = load_horse_race_history(JVLINK_DATA_OUT / "horse_race_history.csv")
    today_for_zensou = entries[["血統登録番号", "馬名", "距離"]].reset_index(drop=True)
    zensou_df_pre = build_zensou_features(today_for_zensou, history_df, kaisai_date)

    zensou_df_pre = zensou_df_pre.replace("連", 1)
    zensou_df_pre = zensou_df_pre.replace("初", 0)

    zensou_df_pre["前走間隔"] = pd.to_numeric(
        zensou_df_pre["前走間隔"], errors="coerce"
    ).astype(float)

    cols = ["前走間隔", "前-2走前間隔", "2-3走前間隔"]
    for c in cols:
        zensou_df_pre[c] = pd.to_numeric(zensou_df_pre[c], errors="coerce")
        zensou_df_pre[c] -= 1

    # 臨戦過程
    l = []
    for i in range(len(zensou_df_pre)):
        if zensou_df_pre.loc[i, "前走間隔"] == 0:
            l.append("連闘")
        elif zensou_df_pre.loc[i, "前走間隔"] >= 9:
            l.append("休明初戦")
        elif zensou_df_pre.loc[i, "前-2走前間隔"] >= 9:
            l.append("休明2走")
        elif zensou_df_pre.loc[i, "2-3走前間隔"] >= 9:
            l.append("休明3走")
        else:
            l.append(np.nan)
    zensou_df_pre["臨戦過程"] = l

    # 距離変遷
    n = []
    for i in range(len(zensou_df_pre)):
        if zensou_df_pre.loc[i, "前走距離"] - zensou_df_pre.loc[i, "距離"] > 0:
            n.append("距離短縮")
        elif zensou_df_pre.loc[i, "前走距離"] - zensou_df_pre.loc[i, "距離"] < 0:
            n.append("距離延長")
        else:
            n.append(np.nan)
    zensou_df_pre["距離変遷"] = n

    zensou_df_pre = zensou_df_pre[["馬名", "臨戦過程", "距離変遷"]]

    # -----------------------------
    # merge
    # -----------------------------

    base_df_preprocessed = base_df_preprocessed.merge(zensou_df_pre, on="馬名", how="left")

    # -----------------------------
    # 出力
    # -----------------------------

    base_df_preprocessed.to_csv(
        DATA_DIR / f"preprocessed_data_{kaisai_date}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
