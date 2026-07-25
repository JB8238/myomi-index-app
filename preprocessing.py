import pandas as pd
import numpy as np
from pathlib import Path
import re

from core.horse_history import load_horse_race_history, build_zensou_features
from core.ck_features import load_ck_data, attach_ck_features, CK_KEY_FIELDS

JVLINK_DATA_OUT = Path(__file__).resolve().parent.parent / "妙味度指数_jvlink" / "data_out"

PLACE_CODE_TO_NAME = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

RACE_KEY = ["開催年", "開催月日", "競馬場コード", "開催回", "開催日目", "レース番号"]

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


def _classify_baba_jotai(turf_code, dirt_code) -> str | None:
    code = turf_code if turf_code not in (None, "", "0") else dirt_code
    return BABA_JOTAI_MAP.get(code)


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
        "芝馬場状態コード", "ダート馬場状態コード",
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
    entries["馬場状態"] = entries.apply(
        lambda r: _classify_baba_jotai(r["芝馬場状態コード"], r["ダート馬場状態コード"]), axis=1
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
    l = []
    ground_pattern = re.compile(r".*良.*")
    for i in range(len(base_pre)):
        baba = base_pre.loc[i, "馬場状態"]
        if pd.isna(baba) or ground_pattern.search(baba):
            l.append(np.nan)
        else:
            if base_pre.loc[i, "種別"] == "芝":
                l.append("芝道悪")
            else:
                l.append("ダ道悪")
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
        if int(kaisai_date) >= 20250215 and int(kaisai_date) <= 20251109:
            peds_df_pre = peds_df.iloc[:, [19, 30]]
        else:
            peds_df_pre = peds_df.iloc[:, [21, 32]]
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
