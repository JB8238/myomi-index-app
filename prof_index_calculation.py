import pandas as pd
import numpy as np
from pathlib import Path
import math
import os

kaisai_date = input("開催日を入力してください (例: 20240601): ")
year = kaisai_date[:4]

INPUT_DIR = Path(".", "list", year)
DATA_DIR = Path(".", "data", year, kaisai_date)
INDEX_DIR = Path(".", "index", kaisai_date)
RESULT_DIR = Path(".", "prof_result", year)

INDEX_DIR.mkdir(exist_ok=True, parents=True)

INPUT_CSV = f"results_prof_index_{kaisai_date}.csv"
OUT_DIR = "target_marks_out"
TARGET_DIR = Path(OUT_DIR , year, kaisai_date)
marks_top5 = ["◎", "○", "▲", "△", "★"]

def all_nan_np(lst):
    """
    NumPy を使って全要素が NaN か判定
    """
    arr = np.array(lst, dtype=float)  # 数値変換できない場合はエラー
    return np.isnan(arr).all()

def split_int_to_17_chunks(n: int):
    """nを最大3つの（<=17）に分割。残りが出る（>51）場合は切り捨て運用にする。"""
    chunks = []
    rem = max(0, int(n))
    for _ in range(3):
        c = min(17, rem)
        chunks.append(c)
        rem -= c
    return chunks  # [m5, m6, m7]

jockey_prof_list_df = pd.read_csv(INPUT_DIR / f"jockey_prof_list_{year}.csv")
sire_prof_list_df = pd.read_csv(INPUT_DIR / f"sire_prof_list_{year}.csv")
trainer_prof_list_df = pd.read_csv(INPUT_DIR / f"trainer_prof_list_{year}.csv")
base_df_prof_preprocessed = pd.read_csv(DATA_DIR / f"preprocessed_data_{kaisai_date}.csv")

base_df_prof_preprocessed = base_df_prof_preprocessed.drop(["距離", "馬場状態"], axis=1)

# 各DFのindexを設定
jockey_prof_list_df = jockey_prof_list_df.set_index("騎手名")
sire_prof_list_df = sire_prof_list_df.set_index("種牡馬名")
trainer_prof_list_df = trainer_prof_list_df.set_index("調教師名")
sire_prof_list_df["重賞"] = sire_prof_list_df["OP"]

# 騎手の各指数を参照
jockey_index_df = []
jockey_columns_name = ["場所", "クラス", "種別", "距離区分", "回り", "道悪判定"]


for i in range(len(base_df_prof_preprocessed)):
    place = base_df_prof_preprocessed.iloc[i]["場所"]
    R = base_df_prof_preprocessed.iloc[i]["R"]
    umaban = base_df_prof_preprocessed.iloc[i]["馬番"]
    horse_name = base_df_prof_preprocessed.iloc[i]["馬名"]
    jockey_name = base_df_prof_preprocessed.iloc[i]["騎手名"]

    # 元コードの姓名幅調整は維持（必要なら）
    if len(jockey_name) == 3:
        jockey_name = jockey_name + "　"
    elif len(jockey_name) == 2:
        jockey_name = jockey_name + "　　"

    jockey_index_list = [place, R, umaban, horse_name, jockey_name]

    # 騎手名がマスタに無ければ、利益度6本を NaN で埋めて次へ
    if jockey_name not in jockey_prof_list_df.index:
        jockey_index_list.extend([np.nan] * len(jockey_columns_name))
        jockey_index_df.append(jockey_index_list)
        continue

    # 6条件ぶん、必ず1本ずつ append（NaNでも）
    for col in jockey_columns_name:
        k = base_df_prof_preprocessed.iloc[i][col]

        # 入力側が欠損なら、この条件の指数は参照しない（NaN）として次へ
        if pd.isna(k):
            jockey_index_list.append(np.nan)
            continue

        # 1条件ごとに安全参照（列が無ければ NaN）
        if k in jockey_prof_list_df.columns:
            m = jockey_prof_list_df.at[jockey_name, k]
        else:
            m = np.nan

        jockey_index_list.append(m)

    jockey_index_df.append(jockey_index_list)

jockey_index_df = pd.DataFrame(jockey_index_df)
jockey_index_df.columns = [
    "場所", "R", "馬番", "馬名", "騎手名", "騎手利益度_1", "騎手利益度_2",
    "騎手利益度_3", "騎手利益度_4", "騎手利益度_5", "騎手利益度_6",
]

# 各行の平均値を算出（NaNは無視して計算）
j_total_list = []
for i in range(len(jockey_index_df)):
    j_num_list = []
    for j in range(5, 11):
        x = jockey_index_df.iloc[i, j]
        j_num_list.append(x)
    if all_nan_np(j_num_list):
        j_total_list.append(np.nan)
    else:
        j_total = np.nansum(j_num_list)
        j_total_list.append(j_total)

jockey_index_df["騎手利益度"] = j_total_list

jockey_index_df["騎手利益度順位"] = jockey_index_df.groupby([
    "場所", "R"
])["騎手利益度"].rank(method="min", ascending=False)
jockey_index_df.to_csv(INDEX_DIR / f"jockey_prof_index_{kaisai_date}.csv", index=False)


# 種牡馬の各指数を参照
sire_index_df = []
sire_columns_name = ["場所", "クラス", "種別", "年齢", "距離区分", "回り", "距離変遷", "道悪判定"]

for i in range(len(base_df_prof_preprocessed)):
    place = base_df_prof_preprocessed.iloc[i]["場所"]
    R = base_df_prof_preprocessed.iloc[i]["R"]
    umaban = base_df_prof_preprocessed.iloc[i]["馬番"]
    horse_name = base_df_prof_preprocessed.iloc[i]["馬名"]
    sire_name = base_df_prof_preprocessed.iloc[i]["種牡馬名"]

    sire_index_list = [place, R, umaban, horse_name, sire_name]

    # 種牡馬名がマスタに無ければ、8個まとめてNaNで埋めて次へ
    if sire_name not in sire_prof_list_df.index:
        sire_index_list.extend([np.nan] * len(sire_columns_name))
        sire_index_df.append(sire_index_list)
        continue

    for col in sire_columns_name:
        k = base_df_prof_preprocessed.iloc[i][col]

        # 入力側が欠損なら、この条件の指数は参照しない（NaN）として次へ
        if pd.isna(k):
            sire_index_list.append(np.nan)
            continue

        # 1条件ごとに安全に参照（無ければNaN）
        if k in sire_prof_list_df.columns:
            m = sire_prof_list_df.at[sire_name, k]
        else:
            m = np.nan

        sire_index_list.append(m)

    sire_index_df.append(sire_index_list)

sire_index_df = pd.DataFrame(sire_index_df)
sire_index_df.columns = [
    "場所", "R", "馬番", "馬名", "種牡馬名", "種牡馬利益度_1",
    "種牡馬利益度_2", "種牡馬利益度_3", "種牡馬利益度_4",
    "種牡馬利益度_5", "種牡馬利益度_6", "種牡馬利益度_7", "種牡馬利益度_8",
]

# 各行の平均値を算出（NaNは無視して計算）
s_total_list = []
for i in range(len(sire_index_df)):
    s_num_list = []
    for j in range(5, 13):
        x = sire_index_df.iloc[i, j]
        s_num_list.append(x)
    if all_nan_np(s_num_list):
        s_total_list.append(np.nan)
    else:
        s_total = np.nansum(s_num_list)
        s_total_list.append(s_total)

sire_index_df["種牡馬利益度"] = s_total_list

sire_index_df["種牡馬利益度順位"] = sire_index_df.groupby([
    "場所", "R"
])["種牡馬利益度"].rank(method="min", ascending=False)
sire_index_df.to_csv(INDEX_DIR / f"sire_prof_index_{kaisai_date}.csv", index=False)


trainer_index_df = []
trainer_columns_name = ["場所", "クラス", "種別", "年齢", "距離区分", "回り", "臨戦過程", "道悪判定"]


for i in range(len(base_df_prof_preprocessed)):
    place = base_df_prof_preprocessed.iloc[i]["場所"]
    R = base_df_prof_preprocessed.iloc[i]["R"]
    umaban = base_df_prof_preprocessed.iloc[i]["馬番"]
    horse_name = base_df_prof_preprocessed.iloc[i]["馬名"]
    trainer_name = base_df_prof_preprocessed.iloc[i]["調教師名"]

    # 元コードの姓名幅調整は維持（必要なら）
    if len(trainer_name) == 3:
        trainer_name = trainer_name + "　"
    elif len(trainer_name) == 2:
        trainer_name = trainer_name + "　　"

    trainer_index_list = [place, R, umaban, horse_name, trainer_name]

    # 調教師名がマスタに無ければ、利益度8本を NaN で埋めて次へ
    if trainer_name not in trainer_prof_list_df.index:
        trainer_index_list.extend([np.nan] * len(trainer_columns_name))
        trainer_index_df.append(trainer_index_list)
        continue

    # 8条件ぶん、必ず1本ずつ append（NaNでも）
    for col in trainer_columns_name:
        k = base_df_prof_preprocessed.iloc[i][col]

        # 入力側が欠損なら、この条件の指数は参照しない（NaN）として次へ
        if pd.isna(k):
            trainer_index_list.append(np.nan)
            continue

        # 1条件ごとに安全参照（列が無ければ NaN）
        if k in trainer_prof_list_df.columns:
            m = trainer_prof_list_df.at[trainer_name, k]
        else:
            m = np.nan

        trainer_index_list.append(m)

    trainer_index_df.append(trainer_index_list)

trainer_index_df = pd.DataFrame(trainer_index_df)
trainer_index_df.columns = [
    "場所", "R", "馬番", "馬名", "調教師名", "調教師利益度_1", "調教師利益度_2",
    "調教師利益度_3", "調教師利益度_4", "調教師利益度_5", "調教師利益度_6",
    "調教師利益度_7", "調教師利益度_8",
]

# 各行の平均値を算出（NaNは無視して計算）
t_total_list = []
for i in range(len(trainer_index_df)):
    t_num_list = []
    for j in range(5, 13):
        x = trainer_index_df.iloc[i, j]
        t_num_list.append(x)
    if all_nan_np(t_num_list):
        t_total_list.append(np.nan)
    else:
        t_total = np.nansum(t_num_list)
        t_total_list.append(t_total)

trainer_index_df["調教師利益度"] = t_total_list

trainer_index_df["調教師利益度順位"] = trainer_index_df.groupby([
    "場所", "R"
])["調教師利益度"].rank(method="min", ascending=False)
trainer_index_df.to_csv(INDEX_DIR / f"trainer_prof_index_{kaisai_date}.csv", index=False)


index_results_df = jockey_index_df.merge(sire_index_df)
index_results_df = index_results_df.merge(trainer_index_df)
index_results_df = index_results_df[[
    "場所", "R", "馬番", "馬名", "騎手利益度", "騎手利益度順位",
    "種牡馬利益度", "種牡馬利益度順位", "調教師利益度", "調教師利益度順位"
]]

index_results_df["総合利益度"] = \
    (index_results_df["騎手利益度"] + index_results_df["種牡馬利益度"] + index_results_df["調教師利益度"]) / 1000

index_results_df["総合利益度順位"] = index_results_df.groupby([
    "場所", "R"
])["総合利益度"].rank(method="min", ascending=False)
index_results_df.to_csv(
    RESULT_DIR / f"results_prof_index_{kaisai_date}.csv",
    index=False, encoding="shift-jis"
)


os.makedirs(TARGET_DIR, exist_ok=True)

df = pd.read_csv(RESULT_DIR / INPUT_CSV, encoding="cp932")
df["総合利益度"] = pd.to_numeric(df["総合利益度"], errors="coerce")

df["M4"] = ""
df["M5"] = ""
df["M6"] = ""
df["M7"] = ""

# レース単位（場所+R）で、総合利益度>0だけを降順ランキング
for (_, _), g in df.groupby(["場所","R"], sort=False):
    pos = g[g["総合利益度"].notna() & (g["総合利益度"] > 0)].copy()
    if pos.empty:
        continue
    pos = pos.sort_values("総合利益度", ascending=False)

    for rank, (idx, row) in enumerate(pos.iterrows(), start=1):
        # 馬印4：上位1～5は◎○▲△★、6位以降は×
        df.loc[idx, "M4"] = marks_top5[rank-1] if rank <= 5 else "×"

        # 馬印5～7：整数部を最大17で分割（例：38 → 17,17,04）
        n = int(math.floor(row["総合利益度"]))
        m5, m6, m7 = split_int_to_17_chunks(n)

        if m5 >= 0: df.loc[idx, "M5"] = f"{m5:02d}"
        if m6 > 0: df.loc[idx, "M6"] = f"{m6:02d}"
        if m7 > 0: df.loc[idx, "M7"] = f"{m7:02d}"

# TARGET取り込み用（馬名,ワークデータ）※ヘッダなし
def export(mark_col, filename):
    out = df[["馬名", mark_col]].rename(columns={mark_col: "ワークデータ"})
    out.to_csv(os.path.join(TARGET_DIR, filename), index=False, header=False, encoding="cp932")

export("M4", f"work_for_mark4_{kaisai_date}.csv")
export("M5", f"work_for_mark5_{kaisai_date}.csv")
export("M6", f"work_for_mark6_{kaisai_date}.csv")
export("M7", f"work_for_mark7_{kaisai_date}.csv")

# 検算用（任意）
df[["場所","R","馬番","馬名","総合利益度","M4","M5","M6","M7"]].to_csv(
    os.path.join(OUT_DIR, "computed_marks_check.csv"),
    index=False, encoding="utf-8"
)