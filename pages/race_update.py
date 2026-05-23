"""
pages/race_update.py

開催当日 変更反映ページ。

STEP 1: netkeiba.com から馬場状態・騎手変更を取得
STEP 2: 検出した変更内容を確認
STEP 3: preprocessed_data を更新して利益度を再計算
STEP 4: 変更ファイルを GitHub へ push
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# プロジェクトルートを絶対パスで確定（Streamlit の __file__ が相対パスになる場合の対策）
_ROOT = Path(__file__).resolve().parent.parent

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.apply_race_changes import apply_changes  # noqa: E402

# スクリプトの絶対パス
_FETCH_SCRIPT = _ROOT / "scripts" / "fetch_netkeiba_changes.py"
_CALC_SCRIPT = _ROOT / "prof_index_calculation.py"

# =========================================================
# ページ設定
# =========================================================
st.set_page_config(page_title="開催当日 変更反映", page_icon="🔄", layout="wide")
st.title("🔄 開催当日 変更反映")
st.caption("馬場状態・騎手変更を netkeiba.com から取得し、利益度を再計算して GitHub へ push します。")

# =========================================================
# Chromium ブラウザの自動インストール
# Streamlit Cloud では pip install playwright だけではブラウザ本体が入らないため、
# アプリ起動時に playwright install chromium を実行する。
# @st.cache_resource でプロセス生存中は一度だけ実行される。
# =========================================================
@st.cache_resource(show_spinner="Chromium を準備中...")
def _ensure_chromium() -> tuple[bool, str]:
    """playwright install chromium を実行してブラウザ本体をインストールする。"""
    r = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return r.returncode == 0, (r.stdout + r.stderr).strip()

# =========================================================
# playwright インストール確認
# =========================================================
def _check_playwright() -> tuple[bool, str]:
    """playwright パッケージが import できるかチェックする。"""
    r = subprocess.run(
        [sys.executable, "-c", "from playwright.async_api import async_playwright; print('OK')"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        return False, r.stderr.strip()
    return True, ""

_pw_ok, _pw_err = _check_playwright()

if not _pw_ok:
    # playwright パッケージ自体が無い → requirements が適用されていない
    st.error(
        "**playwright がインストールされていません。**\n\n"
        "GitHub へ push した後、Streamlit Cloud の管理画面で "
        "**Reboot app** を実行してください。"
    )
    st.code("pip install playwright\nplaywright install chromium", language="bash")
    with st.expander("詳細エラー"):
        st.code(_pw_err, language="text")
    st.info(f"使用中の Python: `{sys.executable}`")
    st.stop()

# playwright はある → Chromium ブラウザ本体を自動インストール
_chromium_ok, _chromium_log = _ensure_chromium()
if not _chromium_ok:
    st.error(
        "**Chromium ブラウザのインストールに失敗しました。**\n\n"
        "packages.txt が正しく設定されているか確認してください。"
    )
    with st.expander("インストールログ"):
        st.code(_chromium_log, language="text")
    st.stop()

# =========================================================
# 開催日入力
# =========================================================
kaisai_date = st.text_input(
    "開催日 (YYYYMMDD)",
    value=datetime.today().strftime("%Y%m%d"),
    placeholder="例: 20260523",
    max_chars=8,
)

if not (len(kaisai_date) == 8 and kaisai_date.isdigit()):
    st.warning("開催日は YYYYMMDD 形式で入力してください。")
    st.stop()

year = kaisai_date[:4]
csv_path = _ROOT / "data" / year / kaisai_date / f"preprocessed_data_{kaisai_date}.csv"
changes_path = _ROOT / "data" / "tmp" / f"netkeiba_changes_{kaisai_date}.json"

if not csv_path.exists():
    st.error(
        f"`preprocessed_data_{kaisai_date}.csv` が見つかりません。"
        "先に `preprocessing.py` を実行してください。"
    )
    st.stop()

st.success(f"`preprocessed_data_{kaisai_date}.csv` を確認しました。", icon="✅")
st.divider()

# =========================================================
# STEP 1: netkeiba から変更情報を取得
# =========================================================
st.subheader("STEP 1: netkeiba から変更情報を取得")

if st.button("netkeiba を取得", type="primary", key="fetch_btn"):
    with st.spinner("netkeiba.com をスクレイピング中... (数分かかる場合があります)"):
        result = subprocess.run(
            [sys.executable, str(_FETCH_SCRIPT), kaisai_date],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(_ROOT),
        )

    if result.returncode != 0:
        st.error("取得エラー")
        # stderr と stdout の両方を表示（どちらにエラー情報があるか不定）
        if result.stderr:
            st.code(result.stderr, language="text")
        if result.stdout:
            with st.expander("標準出力"):
                st.code(result.stdout, language="text")
        # 診断情報
        with st.expander("診断情報"):
            st.text(f"Python: {sys.executable}")
            st.text(f"スクリプト: {_FETCH_SCRIPT}")
            st.text(f"作業ディレクトリ: {_ROOT}")
    else:
        st.success("取得完了")
        if result.stdout:
            with st.expander("実行ログ"):
                st.code(result.stdout, language="text")
        # セッションの古い changes をクリア
        st.session_state.pop("changes", None)
        st.session_state.pop("calc_done", None)

# JSON が存在すればセッションに読み込む
if changes_path.exists() and "changes" not in st.session_state:
    with open(changes_path, encoding="utf-8") as f:
        st.session_state["changes"] = json.load(f)

# =========================================================
# STEP 2: 変更内容の確認
# =========================================================
if "changes" in st.session_state:
    changes: dict = st.session_state["changes"]

    if changes.get("error"):
        st.error(changes["error"])
        st.stop()

    fetched_at = changes.get("fetched_at", "不明")
    track_changes: list[dict] = changes.get("track_changes", [])
    jockey_changes: list[dict] = changes.get("jockey_changes", [])

    st.divider()
    st.subheader("STEP 2: 変更内容の確認")
    st.caption(f"取得日時: {fetched_at}")

    if not track_changes and not jockey_changes:
        st.info("変更は検出されませんでした。")
    else:
        col_track, col_jockey = st.columns(2)

        with col_track:
            st.markdown(f"**馬場状態の変更: {len(track_changes)} 件**")
            if track_changes:
                df_track = pd.DataFrame(track_changes).rename(
                    columns={
                        "旧馬場状態": "現在",
                        "新馬場状態": "変更後",
                    }
                )
                st.dataframe(df_track, use_container_width=True, hide_index=True)
            else:
                st.write("なし")

        with col_jockey:
            st.markdown(f"**騎手の変更: {len(jockey_changes)} 件**")
            if jockey_changes:
                df_jockey = pd.DataFrame(jockey_changes).rename(
                    columns={
                        "旧騎手名": "現在",
                        "新騎手名": "変更後",
                    }
                )
                st.dataframe(df_jockey, use_container_width=True, hide_index=True)
            else:
                st.write("なし")

        st.divider()

        # =========================================================
        # STEP 3: 変更を適用して利益度を再計算
        # =========================================================
        st.subheader("STEP 3: 変更を適用・利益度を再計算")

        if st.button("変更を適用して利益度を再計算", type="primary", key="apply_btn"):
            # CSV に変更を適用
            with st.spinner("preprocessed_data を更新中..."):
                apply_result = apply_changes(kaisai_date, changes)

            if not apply_result["success"]:
                st.error(f"適用エラー: {apply_result['error']}")
                st.stop()

            st.success(
                f"CSV 更新完了 "
                f"（馬場状態: {apply_result['applied_track']} 行, "
                f"騎手: {apply_result['applied_jockey']} 行）"
            )
            with st.expander("バックアップ"):
                st.code(apply_result["backup_path"], language="text")

            # prof_index_calculation.py を実行
            with st.spinner("利益度を再計算中..."):
                calc_result = subprocess.run(
                    [sys.executable, str(_CALC_SCRIPT), kaisai_date],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(_ROOT),
                )

            if calc_result.returncode != 0:
                st.error("利益度計算エラー")
                if calc_result.stderr:
                    st.code(calc_result.stderr, language="text")
                if calc_result.stdout:
                    with st.expander("標準出力"):
                        st.code(calc_result.stdout, language="text")
            else:
                st.success("利益度再計算完了")
                st.session_state["calc_done"] = True
                if calc_result.stdout:
                    with st.expander("計算ログ"):
                        st.code(calc_result.stdout, language="text")

        # =========================================================
        # STEP 4: GitHub へ push
        # =========================================================
        if st.session_state.get("calc_done"):
            st.divider()
            st.subheader("STEP 4: GitHub へ push")

            push_files = [
                f"data/{year}/{kaisai_date}/preprocessed_data_{kaisai_date}.csv",
                f"index/{kaisai_date}/",
                f"prof_result/{year}/results_prof_index_{kaisai_date}.csv",
            ]
            st.caption("push 対象ファイル:")
            for f in push_files:
                st.code(f, language="text")

            if st.button("GitHub へ push", type="primary", key="push_btn"):
                cwd = str(_ROOT)
                push_logs: list[str] = []
                push_error = False

                # git add
                for f in push_files:
                    r = subprocess.run(
                        ["git", "add", f],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        cwd=cwd,
                    )
                    if r.returncode != 0:
                        push_logs.append(f"[warn] git add {f}: {r.stderr.strip()}")

                # git commit
                commit_msg = (
                    f"update: {kaisai_date} 開催中変更反映 "
                    f"(馬場:{len(track_changes)}件, 騎手:{len(jockey_changes)}件)"
                )
                commit_r = subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=cwd,
                )
                push_logs.append(f"git commit: {commit_r.stdout.strip()}")

                if commit_r.returncode != 0:
                    if "nothing to commit" in (commit_r.stdout + commit_r.stderr):
                        st.info("コミットする変更がありませんでした（すでに最新の状態です）。")
                    else:
                        st.error("コミットエラー")
                        st.code(commit_r.stderr, language="text")
                        push_error = True

                if not push_error:
                    push_r = subprocess.run(
                        ["git", "push"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        cwd=cwd,
                    )
                    push_logs.append(
                        f"git push: {(push_r.stdout + push_r.stderr).strip()}"
                    )

                    if push_r.returncode != 0:
                        st.error("push エラー")
                        st.code(push_r.stderr, language="text")
                    else:
                        st.success("GitHub へのpush 完了！")
                        st.balloons()

                with st.expander("git ログ"):
                    st.code("\n".join(push_logs), language="text")
