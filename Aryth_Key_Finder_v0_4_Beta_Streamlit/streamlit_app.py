from __future__ import annotations

import gc
import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from key_analyzer import analyze_audio_file


APP_NAME = "Aryth Key Finder"
APP_VERSION = "v0.4 Beta"

SUPPORTED_EXTENSIONS = ["mp3", "wav", "m4a", "flac"]
MAX_UPLOAD_MB = 300


KEY_GUIDE = """
### Major / MinorとCamelotの対応

- **Major（長調）＝ Camelot B**
- **Minor（短調）＝ Camelot A**
- 例：`G Major（長調） = 9B`
- 例：`E Minor（短調） = 9A`

同じ番号のAとBは相対調の関係です。構成音が近いため、
自動解析ではMajorとMinorの判別が曖昧になる場合があります。

| 番号 | Minor（短調）/ A | Major（長調）/ B |
|---:|---|---|
| 1 | G♯ Minor / 1A | B Major / 1B |
| 2 | D♯ Minor / 2A | F♯ Major / 2B |
| 3 | A♯ Minor（B♭ Minor）/ 3A | C♯ Major（D♭ Major）/ 3B |
| 4 | F Minor / 4A | G♯ Major（A♭ Major）/ 4B |
| 5 | C Minor / 5A | D♯ Major（E♭ Major）/ 5B |
| 6 | G Minor / 6A | A♯ Major（B♭ Major）/ 6B |
| 7 | D Minor / 7A | F Major / 7B |
| 8 | A Minor / 8A | C Major / 8B |
| 9 | E Minor / 9A | G Major / 9B |
| 10 | B Minor / 10A | D Major / 10B |
| 11 | F♯ Minor / 11A | A Major / 11B |
| 12 | C♯ Minor / 12A | E Major / 12B |
"""


CUSTOM_CSS = """
<style>
:root {
    --aryth-accent: #20b26b;
    --aryth-accent-soft: rgba(32, 178, 107, 0.13);
    --aryth-border: rgba(32, 178, 107, 0.34);
}

.block-container {
    max-width: 1180px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

.aryth-hero {
    padding: 1.35rem 1.5rem;
    border: 1px solid var(--aryth-border);
    border-radius: 18px;
    background:
        radial-gradient(
            circle at top right,
            rgba(32, 178, 107, 0.20),
            transparent 38%
        ),
        rgba(128, 128, 128, 0.035);
    margin-bottom: 1rem;
}

.aryth-title {
    margin: 0;
    line-height: 1.15;
    font-size: clamp(2rem, 5vw, 3.15rem);
}

.aryth-subtitle {
    margin: 0.55rem 0 0;
    opacity: 0.88;
    font-size: 1.04rem;
}

.aryth-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.48rem;
    margin-top: 0.95rem;
}

.aryth-badge {
    display: inline-block;
    padding: 0.3rem 0.68rem;
    border-radius: 999px;
    background: var(--aryth-accent-soft);
    border: 1px solid var(--aryth-border);
    font-size: 0.86rem;
}

.aryth-notice {
    padding: 0.82rem 1rem;
    border-left: 4px solid var(--aryth-accent);
    border-radius: 9px;
    background: var(--aryth-accent-soft);
    margin: 0.8rem 0 1.15rem;
}

.aryth-footer {
    margin-top: 2.2rem;
    opacity: 0.78;
    font-size: 0.88rem;
}

div[data-testid="stForm"] {
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 15px;
    padding: 1rem;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-radius: 12px;
    padding: 0.65rem 0.8rem;
}

.stButton button,
.stFormSubmitButton button {
    border-radius: 10px;
    font-weight: 700;
}
</style>
"""


class StreamlitProgress:
    """key_analyzerのコールバック形式をStreamlitへ変換する。"""

    def __init__(self) -> None:
        self._bar = st.progress(0, text="解析の準備中…")
        self._last_value = 0.0

    def __call__(
        self,
        value: float,
        desc: str | None = None,
        **_: Any,
    ) -> None:
        safe_value = max(self._last_value, min(1.0, float(value)))
        self._last_value = safe_value
        self._bar.progress(
            int(round(safe_value * 100)),
            text=desc or "解析中…",
        )

    def finish(self) -> None:
        self._bar.progress(100, text="解析完了！")
        time.sleep(0.25)
        self._bar.empty()

    def fail(self) -> None:
        self._bar.empty()


def initialize_state() -> None:
    defaults = {
        "analysis_result": None,
        "analysis_signature": None,
        "analysis_filename": None,
        "analysis_elapsed": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def file_signature(
    file_bytes: bytes,
    sensitivity: str,
    notation: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(file_bytes)
    digest.update(sensitivity.encode("utf-8"))
    digest.update(notation.encode("utf-8"))
    digest.update(APP_VERSION.encode("utf-8"))
    return digest.hexdigest()


def write_temporary_audio(
    file_bytes: bytes,
    original_name: str,
) -> str:
    suffix = Path(original_name).suffix.lower()

    if suffix not in {".mp3", ".wav", ".m4a", ".flac"}:
        raise ValueError("対応していないファイル形式です。")

    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        prefix="aryth_key_finder_",
        delete=False,
    ) as temporary_file:
        temporary_file.write(file_bytes)
        temporary_file.flush()
        return temporary_file.name


def analyze_uploaded_audio(
    file_bytes: bytes,
    filename: str,
    sensitivity: str,
    notation: str,
) -> tuple[Any, ...]:
    temporary_path: str | None = None
    progress = StreamlitProgress()

    try:
        temporary_path = write_temporary_audio(
            file_bytes=file_bytes,
            original_name=filename,
        )

        result = analyze_audio_file(
            audio_path=temporary_path,
            sensitivity=sensitivity,
            notation=notation,
            progress_callback=progress,
        )
        progress.finish()
        return result

    except Exception:
        progress.fail()
        raise

    finally:
        if temporary_path:
            try:
                os.remove(temporary_path)
            except OSError:
                pass

        gc.collect()


def render_empty_state() -> None:
    st.info(
        "左側で曲と設定を選び、"
        "「キーと転調を解析する」を押してください。"
    )

    st.markdown(
        """
**表示例**

- `G Major（長調） / 9B`
- `E Minor（短調） / 9A`
"""
    )


def render_summary_metrics(details: dict[str, Any]) -> None:
    columns = st.columns(4)

    columns[0].metric(
        "推定主調",
        details.get("推定主調", "—"),
    )
    columns[1].metric(
        "参考信頼度",
        (
            f'{details.get("参考信頼度", "—")}%'
            if details.get("参考信頼度") is not None
            else "—"
        ),
    )
    columns[2].metric(
        "検出した転調",
        f'{details.get("検出転調数", 0)}回',
    )
    columns[3].metric(
        "キーへの復帰",
        f'{details.get("キーへの復帰回数", 0)}回',
    )


def render_dataframe(
    dataframe: pd.DataFrame,
    empty_message: str,
) -> None:
    if dataframe is None or dataframe.empty:
        st.info(empty_message)
        return

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )


def render_results() -> None:
    stored = st.session_state.analysis_result

    if not stored:
        render_empty_state()
        return

    (
        result_markdown,
        segment_table,
        boundary_table,
        candidate_table,
        details,
    ) = stored

    st.caption(
        f'解析ファイル：{st.session_state.analysis_filename}'
        + (
            f'　／　処理時間：{st.session_state.analysis_elapsed:.1f}秒'
            if st.session_state.analysis_elapsed is not None
            else ""
        )
    )

    render_summary_metrics(details)
    st.markdown(result_markdown, unsafe_allow_html=True)

    tab_segments, tab_boundaries, tab_candidates, tab_internal = st.tabs(
        [
            "区間ごとのキー",
            "転調位置",
            "主調候補",
            "内部情報",
        ]
    )

    with tab_segments:
        render_dataframe(
            segment_table,
            "区間情報はありません。",
        )

    with tab_boundaries:
        render_dataframe(
            boundary_table,
            "明確な転調境界は検出されませんでした。",
        )

    with tab_candidates:
        render_dataframe(
            candidate_table,
            "候補ランキングを表示できませんでした。",
        )

    with tab_internal:
        st.json(details)


st.set_page_config(
    page_title=f"{APP_NAME} {APP_VERSION}",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_state()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    f"""
<section class="aryth-hero">
  <h1 class="aryth-title">{APP_NAME}</h1>
  <p class="aryth-subtitle">
    楽曲のキー、Camelot表記、複数回の転調、
    元キーへの復帰を自動解析する実験版ツールです。
  </p>
  <div class="aryth-badges">
    <span class="aryth-badge">{APP_VERSION}</span>
    <span class="aryth-badge">Major＝長調＝Camelot B</span>
    <span class="aryth-badge">Minor＝短調＝Camelot A</span>
    <span class="aryth-badge">複数転調・復帰対応</span>
    <span class="aryth-badge">BPMなし・キー特化</span>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="aryth-notice">
<strong>対応目安：</strong>
MP3 / WAV / M4A / FLAC、12秒以上20分以内<br>
<strong>おすすめ：</strong>
最初は「標準」。短い転調や元キーへの復帰を探すときは
「高め」を試してください。
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("解析設定")

    with st.form(
        "analysis_form",
        clear_on_submit=False,
        border=False,
    ):
        uploaded_file = st.file_uploader(
            "解析する曲",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=False,
            help=(
                f"最大{MAX_UPLOAD_MB}MB。"
                "アップロード音源は一時処理後に削除されます。"
            ),
        )

        sensitivity = st.radio(
            "転調検出の感度",
            [
                "低め（誤検出を抑える）",
                "標準",
                "高め（短い転調も拾う）",
            ],
            index=1,
            help=(
                "誤検出が多い場合は低め、"
                "短い転調や復帰を探す場合は高め。"
            ),
        )

        notation = st.radio(
            "異名同音の表記",
            ["♯優先", "♭優先"],
            index=0,
            horizontal=True,
            help="F♯ / G♭のような表記方法を選びます。",
        )

        submitted = st.form_submit_button(
            "キーと転調を解析する",
            type="primary",
            use_container_width=True,
        )

    if uploaded_file is not None:
        st.audio(uploaded_file)

    st.divider()

    with st.expander("Major / Minor・Camelot対応表"):
        st.markdown(KEY_GUIDE)

    with st.expander("解析の注意点"):
        st.markdown(
            """
- 自動判定のため、必ず正解するわけではありません。
- 属調・下属調方向のコード進行を転調として検出する場合があります。
- 相対長調と相対短調は、判別が曖昧になることがあります。
- 借用和音やモード変化を転調と解釈する場合があります。
- コード感の薄い曲やドラム中心の曲では精度が下がります。
"""
        )

if submitted:
    if uploaded_file is None:
        st.error("解析する音声ファイルを選択してください。")
    else:
        file_bytes = uploaded_file.getvalue()
        upload_size_mb = len(file_bytes) / (1024 * 1024)

        if upload_size_mb > MAX_UPLOAD_MB:
            st.error(
                f"ファイルが大きすぎます。"
                f"{MAX_UPLOAD_MB}MB以下にしてください。"
            )
        else:
            signature = file_signature(
                file_bytes=file_bytes,
                sensitivity=sensitivity,
                notation=notation,
            )

            if (
                signature == st.session_state.analysis_signature
                and st.session_state.analysis_result is not None
            ):
                st.toast("同じ設定の解析結果を再表示しました。")
            else:
                started_at = time.perf_counter()

                try:
                    with st.status(
                        "音声を解析しています…",
                        expanded=True,
                    ) as status:
                        result = analyze_uploaded_audio(
                            file_bytes=file_bytes,
                            filename=uploaded_file.name,
                            sensitivity=sensitivity,
                            notation=notation,
                        )
                        status.update(
                            label="解析が完了しました！",
                            state="complete",
                            expanded=False,
                        )

                    st.session_state.analysis_result = result
                    st.session_state.analysis_signature = signature
                    st.session_state.analysis_filename = uploaded_file.name
                    st.session_state.analysis_elapsed = (
                        time.perf_counter() - started_at
                    )
                    st.toast("解析完了！")

                except Exception as error:
                    st.session_state.analysis_result = None
                    st.session_state.analysis_signature = None
                    st.error(
                        "解析できませんでした。\n\n"
                        f"**原因：** {error}"
                    )

st.header("解析結果")
render_results()

st.markdown(
    """
<div class="aryth-footer">
<hr>
アップロード音源は解析処理にのみ使用し、
一時ファイルは解析終了後に削除します。
権利を有する音源、または解析が許可されている音源を使用してください。
</div>
""",
    unsafe_allow_html=True,
)
