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

# プロジェクトルートを import パスに追加（scripts モジュール参照用）
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.apply_race_changes import apply_changes  # noqa: E402

# =========================================================
# ページ設定
# =========================================================
st.set_page_config(page_title="開催当日 変更反映", page_icon="🔄", layout="wide")
st.title("🔄 開催当日 変更反映")
st.caption("馬場状態・騎手変更を netkeiba.com から取得し、利益度を再計算して GitHub へ push します。")

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
csv_path = Path(f"data/{year}/{kaisai_date}/preprocessed_data_{kaisai_date}.csv")
changes_path = Path(f"data/tmp/netkeiba_changes_{kaisai_date}.json")

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
            [sys.executable, "scripts/fetch_netkeiba_changes.py", kaisai_date],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(_ROOT),
        )
    if result.returncode != 0:
        st.error("取得エラー")
        st.code(result.stderr or result.stdout, language="text")
    else:
        st.success("取得完了")
        # セッションの古い changes をクリア
        st.session_state.pop("changes", None)
        st.session_state.pop("calc_done", None)

    if result.stdout:
        with st.expander("実行ログ"):
            st.code(result.stdout, language="text")

# JSON が存在すればセッションに読み込む
if changes_path.exists() and "changes" not in st.session_state:
    with open(changes_path, encoding="utf-8") as f:
        st.session_state["changes"] = json.load(f)

# =========================================================
# STEP 2: 変更内容の確認
# =========================================================
if "changes" in st.session_state:
    changes: dict = st.session_state["changes"]

    # エラーがある場合は表示して中断
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
                        "場所": "場所",
                        "種別": "種別",
                        "旧馬場状態": "現在",
                        "新馬場状態": "変更後",
                        "影響行数": "影響行数",
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
                        "場所": "場所",
                        "R": "R",
                        "馬番": "馬番",
                        "馬名": "馬名",
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
                    [sys.executable, "prof_index_calculation.py", kaisai_date],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(_ROOT),
                )

            if calc_result.returncode != 0:
                st.error("利益度計算エラー")
                st.code(calc_result.stderr or calc_result.stdout, language="text")
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
                        push_logs.append(f"git add {f}: {r.stderr}")

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
                    # git push
                    push_r = subprocess.run(
                        ["git", "push"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        cwd=cwd,
                    )
                    push_logs.append(f"git push: {push_r.stdout.strip() or push_r.stderr.strip()}")

                    if push_r.returncode != 0:
                        st.error("push エラー")
                        st.code(push_r.stderr, language="text")
                    else:
                        st.success("GitHub へのpush 完了！")
                        st.balloons()

                with st.expander("git ログ"):
                    st.code("\n".join(push_logs), language="text")
