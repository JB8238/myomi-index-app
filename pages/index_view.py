import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(".", "prof_result")  # 指定フォルダ（自動読み込み）
KAKO_DIR = Path(".", "kako_data")    # 過去レースのCSVがあれば（任意）
PREP_DIR = Path(".", "data")
MERGED_RETURN_PATH = Path("./data/return_data_merged.csv")
BUY_WIN_FULL_PATH = Path("./data/buy_conditions_full_win.csv")
BUY_PLC_FULL_PATH = Path("./data/buy_conditions_full_place.csv")

RESULT_COLS = [
    "人気",
    "単オッズ",
    "複勝オッズ下",
    "上",
    "単勝",
    "複勝",
    "的中種別",
    "馬連配当",
]

# --------------------------------------
# prof_result 全CSVから「前走総合利益度」を引くための履歴テーブル
# --------------------------------------
@st.cache_data(show_spinner="📚 prof_result 履歴を整備しています…")
def build_prof_history(data_dir_str: str) -> pd.DataFrame:
    data_dir = Path(data_dir_str)
    files = sorted([p for p in data_dir.rglob("*.csv") if p.is_file()])
    rows = []
    for p in files:
        date_int = extract_yyyymmdd_from_name(p.name)
        if date_int is None:
            continue
        mtime = p.stat().st_mtime    # 「日時」に近い情報として更新時刻を併用
        try:
            df0 = pd.read_csv(str(p), encoding="cp932")
        except Exception:
            # 文字コード差分がある場合に備えて最低限
            df0 = pd.read_csv(str(p), encoding="cp932", errors="ignore")
        
        if "馬名" not in df0.columns or "総合利益度" not in df0.columns:
            continue

        # 必要最低限の列に絞る（無い列は落ちないように）
        keep = [c for c in ["場所", "R", "馬番", "馬名", "総合利益度"] if c in df0.columns]
        df1 = df0[keep].copy()
        df1["__date"] = int(date_int)
        df1["__mtime"] = float(mtime)
        df1["__file"] = p.name

        # 型を整える
        if "R" in df1.columns:
            df1["R"] = pd.to_numeric(df1["R"], errors="coerce")
        if "馬番" in df1.columns:
            df1["馬番"] = pd.to_numeric(df1["馬番"], errors="coerce")
        df1["総合利益度"] = pd.to_numeric(df1["総合利益度"], errors="coerce")
        df1["馬名"] = df1["馬名"].astype(str)

        rows.append(df1)
    
    if not rows:
        return pd.DataFrame(columns=["馬名", "総合利益度", "__date", "__mtime", "__file"])
    
    hist = pd.concat(rows, ignore_index=True)
    # 馬名が空の行は除外
    hist = hist[hist["馬名"].notna() & (hist["馬名"].astype(str).str.strip() != "")]
    return hist

@st.cache_data(show_spinner="📥 preprocessed_data 読み込み中…")
def load_preprocessed_for_race(prep_dir: Path, target_date: int) -> pd.DataFrame:
    files = sorted(prep_dir.rglob("preprocessed_data_*.csv"))
    rows = []

    for p in files:
        d = extract_yyyymmdd_from_name(p.name)
        if d != target_date:
            continue

        df0 = pd.read_csv(p, encoding="utf-8")
        df0.columns = [str(c).strip() for c in df0.columns]

        if not {"場所", "R", "レースレベル"}.issubset(df0.columns):
            continue

        tmp = df0[["場所", "R", "レースレベル"]].copy()
        tmp["開催日"] = d

        tmp["場所"] = (
            tmp["場所"].astype(str)
            .str.replace("\u3000", " ")
            .str.strip()
        )
        tmp["R"] = pd.to_numeric(tmp["R"], errors="coerce")
        tmp["レースレベル"] = tmp["レースレベル"].astype(str).str.strip()

        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["開催日", "場所", "R", "レースレベル"])

    return pd.concat(rows, ignore_index=True)


# クエリパラメータから開催/Rを取得
params = st.query_params
place = params.get("place")
race_no = params.get("race")
date_param = params.get("date")

if place is not None:
    st.session_state["place_select"] = place
if race_no is not None:
    st.session_state["race_select"] = int(race_no)
if date_param is not None:
    st.session_state["date_select"] = int(date_param)

place = st.session_state.get("place_select")
race_no = st.session_state.get("race_select")
race_date = st.session_state.get("date_select")

# --------------------------------------
# Homeに戻るリンク（ページ上部に表示）
# --------------------------------------
back_params = {}
if race_date is not None:
    back_params["date"] = race_date

st.page_link(
    "app.py",
    label="← 開催レース一覧へ戻る",
    icon="🏠",
    use_container_width=True,
    query_params=back_params,
)
st.divider()

# --------------------------------------------
# 1) ファイル探索＆最新判定（YYYYMMDD最大）
# --------------------------------------------
def list_csv_files(data_dir) -> list:
    """指定フォルダ内のCSVファイルをリストアップ"""
    return sorted([p for p in data_dir.rglob("*.csv") if p.is_file()])

def pick_latest_kako_csv(kako_dir: Path, target_date: int | None = None) -> Path | None:
    """kako_data_YYYYMMDD.csv を探し、target_date一致を優先、なければ日付最大を返す"""
    files = sorted([p for p in kako_dir.rglob("kako_data_*.csv") if p.is_file()])
    if not files:
        return None
    dated = []
    for f in files:
        d = extract_yyyymmdd_from_name(f.name)
        if d is not None:
            dated.append((d, f))
    if not dated:
        return None
    if target_date is not None:
        same = [t for t in dated if t[0] == target_date]
        if same:
            same.sort(key=lambda x: x[1].name)
            return same[-1][1]
    dated.sort(key=lambda x: (x[0], x[1].name))
    return dated[-1][1]

@st.cache_data(show_spinner="📥 kako_data を読み込んでいます…")
def load_kako_csv(path_str: str) -> pd.DataFrame:
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path_str, header=None, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path_str, header=None, encoding="cp932", errors="ignore")

def compute_prev_total(df: pd.DataFrame) -> pd.Series:
    block = df.iloc[:, 34:37].apply(pd.to_numeric, errors="coerce")
    return block.sum(axis=1, min_count=1)

def extract_yyyymmdd_from_name(filename: str) -> int | None:
    candidates = re.findall(r"\d{8}", filename)
    valid = []
    for s in candidates:
        try:
            datetime.strptime(s, "%Y%m%d")
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
    dated.sort(key=lambda x: (x[0], x[1].name))
    return dated[-1][1]

# --------------------------------------------
# 2) CSV読み込み（キャッシュ）
# --------------------------------------------
@st.cache_data(show_spinner="📥 CSVを読み込んでいます…")
def load_csv(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str, encoding="cp932")
    return df

# ---------------------------
# 3) 型整形・欠損整備
# ---------------------------
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
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
    return df

# ---------------------------
# 4) フィルタ
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
    page_title="指数ビュー",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="auto",  # 小さい画面ではサイドバーが隠れる挙動が説明されています[10](https://www.johal.in/streamlit-columns-layout-responsive-design-2025/)
)

st.title("🏇 競馬 指数ビューア")
st.caption("prof_result/フォルダ内のCSVを自動読み込み（ファイル名YYYYMMDD最大を最新として選択）")


# --- サイドバー：データソース ---
st.sidebar.header("データ設定")

files = list_csv_files(DATA_DIR)
if race_date is not None:
    files_same_date = [
        p for p in files
        if extract_yyyymmdd_from_name(p.name) == race_date
    ]
    latest = pick_latest_by_filename(files_same_date) if files_same_date else None
else:
    latest = pick_latest_by_filename(files)
    
if latest is None:
    st.error("prof_result/ に YYYYMMDD を含むCSVが見つかりません。")
    st.stop()

qp = st.query_params
csv_name = qp.get("csv")
files = list_csv_files(DATA_DIR)
selected_file = None
if csv_name:
    for p in files:
        if p.name == csv_name:
            selected_file = p
            break
if selected_file is None:
    selected_file = st.session_state.get("selected_prof_csv")
if selected_file is None:
    selected_file = pick_latest_by_filename(files)

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
    format_func=lambda p: p.name,
    key="file_select",
)

# 再読み込み（キャッシュクリア）
# st.cache_data.clear() で全cacheをクリアできることがドキュメントにあります[1](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)[6](https://js2iiu.com/2024/08/28/streamlit-01-cache/)
if st.sidebar.button("🔄 再読み込み（キャッシュクリア）"):
    st.cache_data.clear()

# --- 読み込み ---
df_raw = load_csv(str(selected_file))
df = normalize_df(df_raw)

# --- レース日（prof_result側のファイル名日付）を取得し、kako_dataもそれに合わせる ---
race_date = extract_yyyymmdd_from_name(Path(selected_file).name)

# --- kako_data 読み込み（リポジトリ内の kako_data/ を参照） ---
df_kako = None
kako_path = None
if KAKO_DIR.exists():
    kako_path = pick_latest_kako_csv(KAKO_DIR, target_date=race_date)
    if kako_path is not None:
        df_kako = load_kako_csv(str(kako_path))

# --- return_data 読み込み（結果CSV） ---
df_return = None
if MERGED_RETURN_PATH.exists():
    df_return = pd.read_csv(MERGED_RETURN_PATH, encoding="utf-8-sig")

    # 列名の正規化（return_data は「Ｒ」全角の可能性があるため）
    df_return.columns = [str(c).strip() for c in df_return.columns]
    rename_map = {"Ｒ": "R"}
    df_return.rename(columns=rename_map, inplace=True)

def normalize_place(s):
    if pd.isna(s):
        return s
    return (
        str(s).replace("\u3000", " ").strip()
    )

if df_return is not None:
    # ---- JOINキーの方を明示的にそろえる ----
    if "場所" in df_return.columns:
        df_return["場所"] = df_return["場所"].apply(normalize_place)
    if "R" in df_return.columns:
        df_return["R"] = pd.to_numeric(df_return["R"], errors="coerce")
    if "馬番" in df_return.columns:
        df_return["馬番"] = pd.to_numeric(df_return["馬番"], errors="coerce")
        
    # df 側もそろえる
    if "場所" in df.columns:
        df["場所"] = df["場所"].apply(normalize_place)
    if "R" in df.columns:
        df["R"] = pd.to_numeric(df["R"], errors="coerce")
    if "馬番" in df.columns:
        df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
    
    if df_return is not None:
        df_ret_day = df_return[df_return["開催日"] == race_date].copy()

        df = df.merge(
            df_ret_day,
            on=["場所", "R", "馬番"],
            how="left",
            validate="m:1"
        )

    # ---- 着順は数値比較するため数値化 ----
    if "着" in df.columns:
        df["着"] = pd.to_numeric(df["着"], errors="coerce")
    
    # ---- return_data の着順を優先して統合 ----
    if "着_result" in df.columns:
        df["着"] = df["着_result"].combine_first(df.get("着"))
    
    if "単勝" in df.columns:
        df["単勝"] = pd.to_numeric(df["単勝"], errors="coerce")
        df["単勝的中"] = df["単勝"] > 0
    else:
        df["単勝的中"] = False
    
    if "複勝" in df.columns:
        df["複勝"] = pd.to_numeric(df["複勝"], errors="coerce")
        df["複勝的中"] = df["複勝"] > 0
    else:
        df["複勝的中"] = False
    
    # 視認性用（任意）
    def _hit_label(row):
        if row.get("単勝的中", False):
            return "単勝"
        if row.get("複勝的中", False):
            return "複勝"
        return "-"
    
    df["的中種別"] = df.apply(_hit_label, axis=1)

# --- Home上部：クイック操作 ---
places = sorted(df["場所"].dropna().unique().tolist()) if "場所" in df.columns else []
races = sorted(df["R"].dropna().unique().astype(int).tolist()) if "R" in df.columns else []

with st.sidebar:
    st.header("条件選択")

    st.caption(
        f"📍 開催: {st.session_state.get('place_select', '-')}"
        f" / 🏁 R: {st.session_state.get('race_select', '-')}"
    )

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

    show_result = st.checkbox(
        "📊 結果を表示", value=True,
        key="show_result",
    )

filtered = apply_filters(
    df=df,
    place=place,
    race_no=race_no,
    pass_only=pass_only,
    pass_threshold=0.0,
    include_missing_total=include_missing
)

# =========================================
# 前走総合利益度：prof_result_*.csv から参照（履歴統合）
# =========================================
history = build_prof_history(str(DATA_DIR))
current_date = extract_yyyymmdd_from_name(Path(selected_file).name)
current_mtime = Path(selected_file).stat().st_mtime

def find_prev_total_for_horse(horse_name: str) -> float:
    # 同一ファイル（現在の selected_file）由来は除外
    h = history[(history["馬名"] == horse_name) & (history["__file"] != Path(selected_file).name)]
    if h.empty or current_date is None:
        return np.nan
    # 「開催日に一番近い日時」= (date, mtime) が現在より前で最大のもの
    h = h[(h["__date"] < current_date) | ((h["__date"] == current_date) & (h["__mtime"] < current_mtime))]
    if h.empty:
        return np.nan
    h = h.sort_values(["__date", "__mtime"])
    v = h.iloc[-1]["総合利益度"]
    return float(v) if pd.notna(v) else np.nan

if "馬名" in filtered.columns:
    filtered = filtered.copy()
    filtered["前走総合利益度"] = filtered["馬名"].astype(str).apply(find_prev_total_for_horse)


# =========================================
# 利益度上昇値・人気乖離算出
# （利益度上昇値：総合利益度の整数部分 - 前走総合利益度）
# （人気乖離：人気 - 総合利益度順位　※人気データがあるときのみ）
# =========================================
if "総合利益度" in filtered.columns:
    cur = pd.to_numeric(filtered["総合利益度"], errors="coerce")
    prev = pd.to_numeric(filtered["前走総合利益度"], errors="coerce")
    filtered["利益度上昇値"] = cur - prev

if "人気" in filtered.columns and "総合利益度順位" in filtered.columns:
    filtered["人気"] = pd.to_numeric(filtered["人気"], errors="coerce")
    filtered["総合利益度順位"] = pd.to_numeric(filtered["総合利益度順位"], errors="coerce")
    filtered["人気乖離"] = filtered["人気"] - filtered["総合利益度順位"]
else:
    filtered["人気乖離"] = np.nan

# =========================================
# レースレベル（preprocessed_data から取得）
# =========================================
race_level = None

if race_date is not None and place is not None and race_no is not None:
    df_prep = load_preprocessed_for_race(PREP_DIR, race_date)

    df_race = df_prep[
        (df_prep["場所"] == place) &
        (df_prep["R"] == race_no)
    ]

    if not df_race.empty:
        race_level = df_race["レースレベル"].mode().iloc[0]

st.metric("レースレベル", race_level if race_level else "—")

st.write(f"**選択中：** 場所={place if place else '-'} / R={race_no if race_no is not None else '-'} / 合格のみ={'ON' if pass_only else 'OFF'}")

@st.cache_data(show_spinner="📥 buy_conditions（analysis出力）を読み込み中…")
def load_buy_conditions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    dfc = pd.read_csv(path, encoding="utf-8-sig")
    dfc.columns = [str(c).strip() for c in dfc.columns]
    # 数値列
    for c in ["up_low", "up_high", "gap_low", "gap_high"]:
        if c in dfc.columns:
            dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
    # bool列
    for c in ["up_include_lowest", "gap_include_lowest"]:
        if c in dfc.columns:
            dfc[c] = dfc[c].astype(bool)
    return dfc

cond_win = load_buy_conditions(BUY_WIN_FULL_PATH)
cond_plc = load_buy_conditions(BUY_PLC_FULL_PATH)

def in_interval(val: float, low: float, high: float, include_lowest: bool) -> bool:
    if pd.isna(val) or pd.isna(low) or pd.isna(high):
        return False
    if include_lowest:
        return (val >= low) and (val <= high)
    return (val > low) and (val <= high)

def judge_row(val_up, val_gap, cond_df_lv: pd.DataFrame):
    """
    戻り値:
    （"✅" or "△" or "", 理由文字列)
    ✅: 上昇値＆人気乖離の両方が条件に合致
    △: 上昇値は合致、ただし人気乖離が未取得（NaN）なので条件付き
    "": 不一致
    """
    if cond_df_lv is None or cond_df_lv.empty or pd.isna(val_up):
        return "", ""

    # 上昇値が一致する候補行を抽出
    cand = cond_df_lv[
        cond_df_lv.apply(
            lambda r: in_interval(val_up, r["up_low"], r["up_high"], r["up_include_lowest"]),
            axis=1
        )
    ]
    if cand.empty:
        return "", ""

    # 人気乖離が無い（事前）なら△で「どの区分なら買いか」を返す
    if pd.isna(val_gap):
        gaps = " / ".join(cand["人気乖離区分"].astype(str).unique().tolist())
        return "△", f"人気乖離が {gaps} なら買い"

    # 人気乖離も一致するか
    ok = cand[
        cand.apply(
            lambda r: in_interval(val_gap, r["gap_low"], r["gap_high"], r["gap_include_lowest"]),
            axis=1
        )
    ]
    if ok.empty:
        gaps = " / ".join(cand["人気乖離区分"].astype(str).unique().tolist())
        return "", f"人気乖離が {gaps} なら買い（現在は不一致）"

    # 一番ROIが高い行を採用して説明に使う（任意）
    best = ok.copy()
    roi_col = "単勝ROI" if "単勝ROI" in ok.columns else ("複勝ROI" if "複勝ROI" in ok.columns else None)
    if roi_col:
        best = best.sort_values(roi_col, ascending=False)
    r0 = best.iloc[0]
    return "✅", f"{r0['上昇値区分']} & {r0['人気乖離区分']} (件数={int(r0['件数'])})"

# 該当Lvだけ条件を使う
if race_level:
    cw = cond_win[cond_win["レースレベル"] == race_level] if not cond_win.empty else pd.DataFrame()
    cp = cond_plc[cond_plc["レースレベル"] == race_level] if not cond_plc.empty else pd.DataFrame()

    filtered = filtered.copy()
    # 単勝
    res = filtered.apply(lambda r: judge_row(r.get("利益度上昇値"), r.get("人気乖離"), cw), axis=1)
    filtered["単勝_条件"] = res.apply(lambda t: t[0])
    filtered["単勝_条件説明"] = res.apply(lambda t: t[1])
    # 複勝
    res2 = filtered.apply(lambda r: judge_row(r.get("利益度上昇値"), r.get("人気乖離"), cp), axis=1)
    filtered["複勝_条件"] = res2.apply(lambda t: t[0])
    filtered["複勝_条件説明"] = res2.apply(lambda t: t[1])

    # レース単位の通知
    if (filtered["単勝_条件"] == "✅").any():
        st.success("✅ このレースは『単勝買い条件（乖離込み）』に合致する馬がいます")
    elif (filtered["単勝_条件"] == "△").any():
        st.warning("△ 単勝条件：上昇値は合致（人気乖離が条件なら買い）")

    if (filtered["複勝_条件"] == "✅").any():
        st.success("✅ このレースは『複勝買い条件（乖離込み）』に合致する馬がいます")
    elif (filtered["複勝_条件"] == "△").any():
        st.warning("△ 複勝条件：上昇値は合致（人気乖離が条件なら買い）")
else:
    st.caption("※ レースレベルが取得できないため、買い条件判定を行いません。")

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
    c for c in [
        "馬番",
        "馬名",
        "総合利益度",
        "前走総合利益度",
        "利益度上昇値",
        "単勝_条件",
        "複勝_条件",
        "総合利益度順位"
    ]
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
            "前走総合利益度": "{:.3f}",
            "利益度上昇値": "{:.3f}",
            "総合利益度順位": "{:.0f}",
        }, na_rep="—")
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
                "前走総合利益度": "{:.3f}",
                "利益度上昇値": "{:.3f}",
                "総合利益度順位": "{:.0f}",
            }, na_rep="—")
            .apply(highlight_row_if_total_ge_17, axis=1)
        )
        st.dataframe(
            styler_full,
            use_container_width=True,
            hide_index=True,
            height=520,
        )

# ======================================
# レース結果ビュー
# ======================================
if st.session_state.get("show_result", False):
    st.subheader("📊 レース結果（オッズ・着順）")

    # 表示可能な結果列だけ抽出
    result_cols = [c for c in RESULT_COLS if c in filtered.columns]

    if "単勝" in filtered.columns:
        hit = filtered[filtered["単勝"] > 0]
        st.metric("単勝 的中数", f"{len(hit)}頭")
    
    if "複勝" in filtered.columns:
        place_hit = filtered[filtered["複勝"] > 0]
        st.metric("複勝 的中数", f"{len(place_hit)}頭")
    
    if "単勝" in filtered.columns:
        st.metric("単勝 回収合計", int(filtered["単勝"].sum()))

    if len(result_cols) == 0:
        st.info("このCSVには結果データが含まれていません。")
    else:
        base_cols = [c for c in ["馬番", "馬名"] if c in filtered.columns]
        result_df = filtered[base_cols + result_cols].copy()

        # 並び順：人気（低いほど上） → 単オッズ（低いほど上）
        if "人気" in result_df.columns:
            result_df["人気"] = pd.to_numeric(result_df["人気"], errors="coerce")
        if "単オッズ" in result_df.columns:
            result_df["単オッズ"] = pd.to_numeric(result_df["単オッズ"], errors="coerce")
        if "人気" in result_df.columns:
            result_df = result_df.sort_values(["人気", "単オッズ"], na_position="last")

        
        styler = (
            result_df.style
            .format({
                "人気": "{:.0f}",
                "単オッズ": "{:.1f}",
                "複勝オッズ下": "{:.1f}",
                "上": "{:.1f}",
                "単勝": "{:.0f}",
                "複勝": "{:.0f}",
                "馬連配当": "{:.0f}",
            }, na_rep="-")
        )

        st.dataframe(
            styler,
            use_container_width=True,
            hide_index=True,
            height=420,
        )