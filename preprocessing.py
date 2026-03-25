import pandas as pd
import numpy as np
from pathlib import Path
import re


def main():

    kaisai_date = input("開催日を入力してください (例: 20240601): ")
    data_pattern = int(input("データパターンを選択してください（1: 確定後データ, 2: 確定前データ）: "))
    year = kaisai_date[:4]

    DATA_DIR = Path(".", "data", year, kaisai_date)
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    KAKO_DATA_DIR = Path("C:/TFJV/TXT/data/kako_data", year)
    BASE_DATA_DIR = Path("C:/TFJV/TXT/data/base_data", year)
    PEDS_DATA_DIR = Path("C:/TFJV/TXT/data/peds_data", year)

    # -----------------------------
    # CSV読み込み
    # -----------------------------

    zensou_df = pd.read_csv(
        KAKO_DATA_DIR / f"kako_data_{kaisai_date}.csv",
        encoding="shift-jis",
        header=None,
    )

    base_df = pd.read_csv(
        BASE_DATA_DIR / f"base_data_{kaisai_date}.csv",
        encoding="shift-jis",
        header=None,
    )

    peds_df = pd.read_csv(
        PEDS_DATA_DIR / f"peds_data_{kaisai_date}.csv",
        encoding="shift-jis",
        header=None,
    )

    # -----------------------------
    # 使用カラム抽出
    # -----------------------------
    if int(kaisai_date) >= 20250215 and int(kaisai_date) <= 20251109:
        zensou_df_pre = zensou_df.iloc[:, [8, 19, 23, 28, 41, 62]]
    else:
        zensou_df_pre = zensou_df.iloc[:, [8, 21, 25, 30, 43, 64]]
    if data_pattern == 1:
        if int(kaisai_date) >= 20250215 and int(kaisai_date) <= 20251109:
            base_df_pre = base_df.iloc[:, [3, 4, 5, 7, 8, 9, 11, 14, 19, 22, 24, 31]]
            peds_df_pre = peds_df.iloc[:, [19, 30]]
        else:
            base_df_pre = base_df.iloc[:, [3, 4, 5, 7, 8, 9, 11, 14, 21, 24, 26, 33]]
            peds_df_pre = peds_df.iloc[:, [21, 32]]
    elif data_pattern == 2:
        if int(kaisai_date) >= 20250215 and int(kaisai_date) <= 20251109:
            base_df_pre = base_df.iloc[:, [3, 4, 5, 7, 8, 9, 11, 14, 19, 22, 24, 29]]
            peds_df_pre = peds_df.iloc[:, [19, 28]]
        else:
            base_df_pre = base_df.iloc[:, [3, 4, 5, 7, 8, 9, 11, 14, 21, 24, 26, 31]]
            peds_df_pre = peds_df.iloc[:, [21, 30]]
    else:
        raise ValueError("不正な値が入力されています。'1'か'2'を入力してください。")

    # -----------------------------
    # 前走データ加工
    # -----------------------------

    zensou_df_pre.columns = [
        "距離",
        "馬名",
        "前走間隔",
        "前走距離",
        "前-2走前間隔",
        "2-3走前間隔",
    ]

    zensou_df_pre = zensou_df_pre.replace("連", 1)
    zensou_df_pre = zensou_df_pre.replace("初", 0)

    zensou_df_pre["前走間隔"] = pd.to_numeric(
        zensou_df_pre["前走間隔"], errors="coerce"
    ).astype(float)

    # int → float変換
    int_cols = zensou_df_pre.select_dtypes(include=["int"]).columns
    zensou_df_pre[int_cols] = zensou_df_pre[int_cols].astype("float")

    # 中○週 → 数値
    cols = ["前走間隔", "前-2走前間隔", "2-3走前間隔"]
    for c in cols:
        zensou_df_pre[c] = pd.to_numeric(zensou_df_pre[c], errors="coerce")
        zensou_df_pre[c] -= 1

    # 臨戦過程
    l = []

    for i in range(len(zensou_df_pre)):

        if zensou_df_pre.loc[i, "前走間隔"] == 0:
            l.append("連闘")

        elif zensou_df_pre.loc[i, "前走間隔"] >= 12:
            l.append("休明初戦")

        elif zensou_df_pre.loc[i, "前-2走前間隔"] >= 12:
            l.append("休明2走")

        elif zensou_df_pre.loc[i, "2-3走前間隔"] >= 12:
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
    # baseデータ加工
    # -----------------------------

    base_pre = base_df_pre.copy()

    base_pre.columns = [
        "場所",
        "R",
        "クラス",
        "種別",
        "距離",
        "馬場状態",
        "レースレベル",
        "馬番",
        "馬名",
        "年齢",
        "騎手名",
        "調教師名",
    ]

    base_pre["クラス"] = base_pre["クラス"].str.replace(r".*未勝利.*", "未勝利", regex=True)
    base_pre["クラス"] = base_pre["クラス"].str.replace(r".*新馬.*", "新馬", regex=True)
    base_pre["クラス"] = base_pre["クラス"].str.replace(
        r".*(１勝|２勝|３勝|1勝|2勝|3勝).*", "自己条件", regex=True
    )
    base_pre["クラス"] = base_pre["クラス"].str.replace(r".*(G1|G2|G3).*", "重賞", regex=True)

    cond = ~base_pre["クラス"].str.contains("新馬|未勝利|自己条件|重賞")

    base_pre.loc[cond, "クラス"] = "OP"

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

    # 年齢
    for i in range(2, 14):

        if i <= 4:
            base_pre["年齢"] = base_pre["年齢"].replace(i, f"{i}歳")

        else:
            base_pre["年齢"] = base_pre["年齢"].replace(i, "5歳以上")

    # 道悪判定
    l = []

    ground_pattern = re.compile(r".*良.*")

    for i in range(len(base_pre)):

        if ground_pattern.search(base_pre.loc[i, "馬場状態"]):
            l.append(np.nan)

        else:

            if base_pre.loc[i, "種別"] == "芝":
                l.append("芝道悪")

            else:
                l.append("ダ道悪")

    base_pre["道悪判定"] = l

    # -----------------------------
    # 血統
    # -----------------------------

    peds_df_pre.columns = ["馬名", "種牡馬名"]

    # -----------------------------
    # merge
    # -----------------------------

    base_df_preprocessed = base_pre.merge(peds_df_pre, on="馬名")
    base_df_preprocessed = base_df_preprocessed.merge(zensou_df_pre, on="馬名")

    # -----------------------------
    # 出力
    # -----------------------------

    base_df_preprocessed.to_csv(
        DATA_DIR / f"preprocessed_data_{kaisai_date}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()