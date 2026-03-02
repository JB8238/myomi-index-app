import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st


DATA_DIR = Path(".","prof_result")   # 指定フォルダ（自動読み込み）

# --------------------------------------------
# 1) ファイル探索＆最新判定（YYYYMMDD最大）
# --------------------------------------------

def list_csv_files(data_dir) -> list:
    """指定フォルダ内のCSVファイルをリストアップ"""
    return sorted([p for p in data_dir.rglob("*.csv") if p.is_file()])

def extract_yyyymmdd_from_name(filename: str) -> int | None:
    """
    ファイル名から YYYYMMDD（8桁）を抽出し、妥当な日付のうち最大を返す。
    例：results_prof_index_20250525.csv -> 20250525
    """
    candidates = re.findall(r"\d{8}", filename)
    valid = []
    for s in candidates:
        try:
            datetime.strptime(s, "%Y%m%d")  # 日付として妥当か確認
            valid.append(int(s))
        except ValueError:
            pass
    return max(valid) if valid else None

def pick_latest_by_filename(files: list[Path]) -> Path | None:
    dated = []
    for f in files:
        d = extract_yyyymmdd_from_name(f.name)
        if d is not None:
            dated.append((d, f))
    if not dated:
        return None
    # 日付最大を最新とする（同日の複数ファイルはファイル名で安定化）
    dated.sort(key=lambda x: (x[0], x[1].name))
    return dated[-1][1]


# --------------------------------------------
# 2) CSV読み込み（キャッシュ）
# --------------------------------------------
@st.cache_data(show_spinner="📥 CSVを読み込んでいます…")

def load_csv(path_str: str) -> pd.DataFrame:
    # ※st.cache_data は戻り値をpickle化してキャッシュします（信頼できるデータ前提）[3](https://github.com/streamlit/docs/blob/main/content/develop/api-reference/caching-and-state/cache-data.md)
    df = pd.read_csv(path_str, encoding="cp932")
    return df


# ---------------------------
# 3) 型整形・欠損整備（あなたのCSVに合わせる）
# ---------------------------
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # 列名の前後空白などがあれば吸収
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # 想定列（あなたのCSVより）[7](https://dev.to/jamesbmour/streamlit-part-6-mastering-layouts-4hci)
    # 場所, R, 馬番, 馬名, ... 総合利益度, 総合利益度順位
    # 数値列は空欄があり得るので to_numeric(errors="coerce") でNaN化
    num_cols = [
        "R", "馬番",
        "騎手利益度", "騎手利益度順位",
        "種牡馬利益度", "種牡馬利益度順位",
        "調教師利益度", "調教師利益度順位",
        "総合利益度", "総合利益度順位",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 表示用：R/馬番は整数に寄せたいがNaNがあるので最後に整形側で対応
    return df


# ---------------------------
# 4) フィルタ（最頻：総合利益度 >= 0）
# ---------------------------
def apply_filters(
    df: pd.DataFrame,
    place: str | None,
    race_no: int | None,
    pass_only: bool = True,
    pass_threshold: float = 0.0,
    include_missing_total: bool = False,
) -> pd.DataFrame:
    out = df.copy()

    if place and "場所" in out.columns:
        out = out[out["場所"] == place]

    if race_no is not None and "R" in out.columns:
        out = out[out["R"] == race_no]

    # 合格条件：総合利益度 >= 0（デフォルト）
    if pass_only and "総合利益度" in out.columns:
        if include_missing_total:
            out = out[(out["総合利益度"].isna()) | (out["総合利益度"] >= pass_threshold)]
        else:
            out = out[out["総合利益度"].notna() & (out["総合利益度"] >= pass_threshold)]

    return out


# ---------------------------
# 5) UI（スマホ向けカード）
# ---------------------------
def render_cards(df: pd.DataFrame):
    if df.empty:
        st.info("条件に合う馬がありません。")
        return

    # 並び：総合利益度 降順（あれば）
    if "総合利益度" in df.columns:
        df = df.sort_values("総合利益度", ascending=False)

    for _, row in df.iterrows():
        umaban = row.get("馬番", "")
        name = row.get("馬名", "")
        total = row.get("総合利益度", None)
        total_rank = row.get("総合利益度順位", None)

        title = f"{'' if pd.isna(umaban) else int(umaban)}  {name}"
        st.markdown(f"### {title}")

        # 主要情報（スマホで横スクロールしない）
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("**総合利益度**")
            st.write("—" if pd.isna(total) else f"{total:.3f}")
        with c2:
            st.write("**総合順位**")
            st.write("—" if pd.isna(total_rank) else f"{int(total_rank)} 位")

        # 詳細は畳む（Expanderは大事）[8](https://kajiblo.com/streamlit-st-cache_data/)[9](https://note.com/pinyo/n/n94a46842bd1d)
        with st.expander("詳細（騎手/種牡馬/調教師）"):
            for label in [
                ("騎手利益度", "騎手利益度順位"),
                ("種牡馬利益度", "種牡馬利益度順位"),
                ("調教師利益度", "調教師利益度順位"),
            ]:
                v = row.get(label[0], None)
                r = row.get(label[1], None)
                st.write(
                    f"- {label[0]}: "
                    f"{'—' if pd.isna(v) else f'{v:.1f}'} / "
                    f"{label[1]}: {'—' if pd.isna(r) else f'{int(r)} 位'}"
                )

        st.divider()


# ---------------------------
# メイン
# ---------------------------
st.set_page_config(
    page_title="競馬 指数ビューア",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="auto",  # 小さい画面ではサイドバーが隠れる挙動が説明されています[10](https://www.johal.in/streamlit-columns-layout-responsive-design-2025/)
)

st.title("🏇 競馬 指数ビューア")
st.caption("prof_result/フォルダ内のCSVを自動読み込み（ファイル名YYYYMMDD最大を最新として選択）")


# --- サイドバー：データソース ---
st.sidebar.header("データ設定")

files = list_csv_files(DATA_DIR)
latest = pick_latest_by_filename(files)

if latest is None:
    st.error("prof_result/ に YYYYMMDD を含むCSVが見つかりません。")
    st.stop()

# 日付降順で並べて選べるように
def file_sort_key(p: Path):
    d = extract_yyyymmdd_from_name(p.name) or 0
    return (d, p.name)

files_sorted = sorted(files, key=file_sort_key, reverse=True)
default_idx = files_sorted.index(latest)

selected_file = st.sidebar.selectbox(
    "読み込みCSV（デフォルトは最新）",
    options=files_sorted,
    index=default_idx,
    format_func=lambda p: p.name
)

# 再読み込み（キャッシュクリア）
# st.cache_data.clear() で全cacheをクリアできることがドキュメントにあります[1](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)[6](https://js2iiu.com/2024/08/28/streamlit-01-cache/)
if st.sidebar.button("🔄 再読み込み（キャッシュクリア）"):
    st.cache_data.clear()

# --- 読み込み ---
df_raw = load_csv(str(selected_file))
df = normalize_df(df_raw)

# # --- Home上部：クイック操作 ---

places = sorted(df["場所"].dropna().unique().tolist()) if "場所" in df.columns else []
races = sorted(df["R"].dropna().unique().astype(int).tolist()) if "R" in df.columns else []

with st.sidebar:
    st.header("条件選択")

    place = st.selectbox(
        "開催（場所）",
        options=places,
        index=0 if places else None,
        key="place_select",
    ) if places else None

    race_no = st.selectbox(
        "R",
        options=races,
        index=0 if races else None,
        key="race_select",
    ) if races else None

    pass_only = st.checkbox(
        "✅ 合格馬だけ（総合利益度 >= 0）",
        value=True,
        key="pass_only",
    )

    include_missing = st.checkbox(
        "総合利益度が欠損の馬も表示する",
        value=False,
        key="include_missing",
    )

filtered = apply_filters(
    df=df,
    place=place,
    race_no=race_no,
    pass_only=pass_only,
    pass_threshold=0.0,
    include_missing_total=include_missing
)

st.write(f"**選択中：** 場所={place if place else '-'} / R={race_no if race_no is not None else '-'} / 合格のみ={'ON' if pass_only else 'OFF'}")

# --- サマリー（任意） ---
st.subheader("サマリー")

# ======================================
# ① モバイル向け：上部カード（KPI）
# ======================================
pass_count = int(len(filtered))
total_count = int(
    len(df[(df["場所"] == place) & (df["R"] == race_no)])
) if place and race_no is not None and "場所" in df.columns and "R" in df.columns else int(len(df))

# 2列にしてスマホで縦積みになりやすい形にする
k1, k2 = st.columns(2)
with k1:
    st.metric("合格頭数", f"{pass_count} / {total_count}")
with k2:
    if "総合利益度" in filtered.columns and not filtered["総合利益度"].dropna().empty:
        st.metric("総合利益度 最大", f"{filtered['総合利益度'].max():.3f}")
    else:
        st.metric("総合利益度 最大", "—")

# 平均も欲しければ（任意）
if "総合利益度" in filtered.columns and not filtered["総合利益度"].dropna().empty:
    st.metric("総合利益度 平均", f"{filtered['総合利益度'].mean():.3f}")

# ======================================
# ②③ モバイル向け：表表示（設定＋列間引き）
# ======================================
st.subheader("結果（モバイル最適）")

# ② 表に出す列を絞る（横スクロールを減らす）
mobile_cols = [
    c for c in ["馬番", "馬名", "総合利益度", "総合利益度順位"]
    if c in filtered.columns
]

if "総合利益度" in filtered.columns:
    filtered_sorted = filtered.sort_values(
        by="総合利益度",
        ascending=False,
        na_position="last",
    )
else:
    filtered_sorted = filtered

if filtered_sorted.empty:
    st.info("条件に合う馬がありません。")
else:
    # 総合利益度 >= 17 の行をハイライト
    def highlight_row_if_total_ge_17(row):
        """
        row: pandas Series（1行）
        条件を満たす行は背景色を付与（列数分のCSS文字列を返す）
        """
        if "総合利益度" in row.index and pd.notna(row["総合利益度"]) and row["総合利益度"] >= 17:
            return ["background-color: rgba(255, 193, 7, 0.25);"] * len(row)
        return [""] * len(row)
    
    df_table = filtered_sorted[mobile_cols] if mobile_cols else filtered_sorted
    styler = (
        df_table.style
        .format({
            "総合利益度": "{:.3f}",
            "総合利益度順位": "{:.0f}",
        })
        .apply(highlight_row_if_total_ge_17, axis=1)
    )

    # ② st.dataframe（モバイル向け設定）は維持しつつ、stylerを渡す
    st.dataframe(
        styler,
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    with st.expander("詳細（全列）"):
        # 詳細側も同じ条件でハイライト
        styler_full = (
            filtered_sorted.style
            .format({
                "騎手利益度": "{:.0f}",
                "騎手利益度順位": "{:.0f}",
                "種牡馬利益度": "{:.0f}",
                "種牡馬利益度順位": "{:.0f}",
                "調教師利益度": "{:.0f}",
                "調教師利益度順位": "{:.0f}",
                "総合利益度": "{:.3f}",
                "総合利益度順位": "{:.0f}",
            })
            .apply(highlight_row_if_total_ge_17, axis=1)
        )
        st.dataframe(
            styler_full,
            use_container_width=True,
            hide_index=True,
            height=520,
        )