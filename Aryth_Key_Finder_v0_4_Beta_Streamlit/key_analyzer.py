"""Aryth Key Finder v0.4 Beta — analysis engine."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, Callable

import librosa
import numpy as np
import pandas as pd


# ============================================================
# Aryth Key Finder v0.4 Beta — analysis engine
# ・転調しない曲では従来の全体判定を維持
# ・複数の時間幅で調性変化を探すハイブリッド検出
# ・転調位置の検出と、転調後キーの推定を分離
# ・相対長短調は同じ調性ファミリーとして扱う
# ============================================================

SAMPLE_RATE = 22_050
HOP_LENGTH = 2_048

# テンポ推定用のオンセット包絡は細かい刻みが要る。
# HOP_LENGTH(2048)だと1秒あたり約10.8フレームしか無く、
# たとえば140や150 BPMを整数ラグで表現できずに桁が飛ぶ。
# 512刻み（約43フレーム/秒）ならこの帯域も十分に分解できる。
ONSET_HOP_LENGTH = 512

BINS_PER_OCTAVE = 36
N_OCTAVES = 7
MAX_DURATION_SECONDS = 20 * 60
MIN_AUDIO_SECONDS = 12.0

# 特徴量はブロックごとに計算する。
# 打楽器成分の抑制（HPSS）は曲の長さに比例してメモリを食い、
# 6分の曲で1GB近くに達する。クロマ自体は1分あたり0.03MBしかないので、
# 短い区間へ切って計算し、クロマだけを残せばメモリは曲の長さに依存しなくなる。
BLOCK_SECONDS = 60.0

# ブロック境界での歪みを避けるための前後ののりしろ。
# C1付近のCQT窓が約1.6秒、CENSの平滑化が約2秒必要なので、
# それらを十分に上回る長さを取る。
BLOCK_PAD_SECONDS = 4.0

# テンポ推定の設定。
# 知覚的に自然なテンポ帯（Moelantsらのいう約120 BPM付近）を
# 中心に置いた対数正規の重みで、倍/半のオクターブ誤差を選び分ける。
TEMPO_MIN_BPM = 40.0
TEMPO_MAX_BPM = 280.0
TEMPO_PRIOR_CENTER = 125.0
TEMPO_PRIOR_SIGMA = 0.9  # log2（オクターブ）単位

# テンポ変化の検出設定。
# 局所テンポを窓で追い、はっきりした変化が十分続いたときだけ区切る。
# 誤検出を強く嫌う方針なので、区間は長め・変化幅は大きめを既定にする。
TEMPO_WINDOW_SECONDS = 12.0
TEMPO_STEP_SECONDS = 3.0
MIN_TEMPO_SEGMENT_SECONDS = 20.0

# 「テンポが変わった」とみなす最小の相対差（5%＝120→126程度）。
TEMPO_CHANGE_RATIO = 0.05

# 2:1（倍/半）の関係はテンポ検出の曖昧さで揺れやすく、
# ハーフタイム／ダブルタイムは同じテンポの取り方違いとみなす。
# そのため比較前にオクターブを揃え、純粋な倍/半は変化として扱わない。

CHANGE_SCAN_STEP_SECONDS = 2.0
CHANGE_CONTEXT_OPTIONS = (12.0, 18.0, 24.0)
CHANGE_GUARD_SECONDS = 1.5
BOUNDARY_REFINEMENT_RADIUS = 8.0
MAX_BOUNDARIES = 8

# 属調・下属調方向の移動はコード進行でも頻繁に現れるため慎重に扱う
FIFTH_SHIFTS = {5, 7}

# 終盤の半音上げなどは比較的明確な転調として拾いやすくする
SEMITONE_SHIFTS = {1, 11}


# ----- キープロファイル -----

KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)

# 別傾向のプロファイルも少量混ぜ、特定の進行への偏りを弱める
TEMPERLEY_MAJOR = np.array(
    [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
    dtype=np.float64,
)
TEMPERLEY_MINOR = np.array(
    [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0],
    dtype=np.float64,
)

# Shaath (KeyFinder) — ポップス／ダンス音源の録音から作られたプロファイル。
# Camelot表記を使う用途とも相性が良いため、本ツールでは主軸に置く。
SHAATH_MAJOR = np.array(
    [6.6, 2.0, 3.5, 2.3, 4.6, 4.0, 2.5, 5.2, 2.4, 3.7, 2.3, 3.4],
    dtype=np.float64,
)
SHAATH_MINOR = np.array(
    [6.5, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 5.2, 4.0, 2.7, 4.3, 3.2],
    dtype=np.float64,
)

# Albrecht & Shanahan — 大規模コーパスから推定されたプロファイル。
# 非和声音に強く、借用和音の多い曲で主音がぶれにくい。
ALBRECHT_MAJOR = np.array(
    [0.238, 0.006, 0.111, 0.006, 0.137, 0.094,
     0.016, 0.214, 0.009, 0.080, 0.008, 0.081],
    dtype=np.float64,
)
ALBRECHT_MINOR = np.array(
    [0.220, 0.006, 0.104, 0.123, 0.019, 0.103,
     0.012, 0.214, 0.062, 0.022, 0.061, 0.052],
    dtype=np.float64,
)

# 低音域では主音・属音・下属音を少し重視する
BASS_MAJOR = np.array(
    [1.00, 0.06, 0.25, 0.06, 0.28, 0.42, 0.06, 0.78, 0.06, 0.22, 0.06, 0.18],
    dtype=np.float64,
)
BASS_MINOR = np.array(
    [1.00, 0.06, 0.25, 0.38, 0.06, 0.34, 0.06, 0.78, 0.32, 0.06, 0.20, 0.16],
    dtype=np.float64,
)


SHARP_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
FLAT_NAMES = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]

MAJOR_CAMELOT = {
    0: "8B", 1: "3B", 2: "10B", 3: "5B", 4: "12B", 5: "7B",
    6: "2B", 7: "9B", 8: "4B", 9: "11B", 10: "6B", 11: "1B",
}
MINOR_CAMELOT = {
    0: "5A", 1: "12A", 2: "7A", 3: "2A", 4: "9A", 5: "4A",
    6: "11A", 7: "6A", 8: "1A", 9: "8A", 10: "3A", 11: "10A",
}

KEY_LABELS = (
    [(tonic, "major") for tonic in range(12)]
    + [(tonic, "minor") for tonic in range(12)]
)


SENSITIVITY_SETTINGS = {
    "低め（誤検出を抑える）": {
        "candidate_threshold": 0.57,
        "minimum_gain": 0.022,
        "minimum_structure": 0.35,
        "minimum_segment_seconds": 20.0,
        "minimum_boundary_gap": 22.0,
        "transition_penalty": 1.55,
        "minimum_scale_change": 0.030,
    },
    "標準": {
        "candidate_threshold": 0.46,
        "minimum_gain": 0.010,
        "minimum_structure": 0.25,
        "minimum_segment_seconds": 14.0,
        "minimum_boundary_gap": 14.0,
        "transition_penalty": 1.20,
        "minimum_scale_change": 0.018,
    },
    "高め（短い転調も拾う）": {
        "candidate_threshold": 0.36,
        "minimum_gain": 0.004,
        "minimum_structure": 0.18,
        "minimum_segment_seconds": 10.0,
        "minimum_boundary_gap": 10.0,
        "transition_penalty": 0.95,
        "minimum_scale_change": 0.009,
    },
}

# 構成音の入れ替わりを「転調あり」とみなす目安。
# クロマは合計1に正規化されているので、1音あたり0.02動けば十分大きい。
SCALE_CHANGE_SCALE = 0.075


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / (values.std() + 1e-12)


def _build_templates(major: np.ndarray, minor: np.ndarray) -> np.ndarray:
    major_z = _standardize(major)
    minor_z = _standardize(minor)
    return np.stack(
        [np.roll(major_z, tonic) for tonic in range(12)]
        + [np.roll(minor_z, tonic) for tonic in range(12)]
    )


KRUMHANSL_TEMPLATES = _build_templates(KRUMHANSL_MAJOR, KRUMHANSL_MINOR)
TEMPERLEY_TEMPLATES = _build_templates(TEMPERLEY_MAJOR, TEMPERLEY_MINOR)
SHAATH_TEMPLATES = _build_templates(SHAATH_MAJOR, SHAATH_MINOR)
ALBRECHT_TEMPLATES = _build_templates(ALBRECHT_MAJOR, ALBRECHT_MINOR)
BASS_TEMPLATES = _build_templates(BASS_MAJOR, BASS_MINOR)

# 長音階の構成音マスク（Cメジャー基準）。転調判定で音階そのものの
# 入れ替わりを確認するために使う。
MAJOR_SCALE_MASK = np.array(
    [True, False, True, False, True, True,
     False, True, False, True, False, True],
)


def scale_membership(tonic: int) -> np.ndarray:
    """長音階（＝相対短調と共通の構成音）の集合を返す。"""
    return np.roll(MAJOR_SCALE_MASK, int(tonic) % 12)


def pitch_name(tonic: int, notation: str) -> str:
    names = SHARP_NAMES if notation == "♯優先" else FLAT_NAMES
    return names[int(tonic) % 12]


def key_name(key_index: int, notation: str) -> str:
    """ユーザー向けのキー表記。英語と日本語を併記する。"""
    tonic, mode = KEY_LABELS[int(key_index)]
    mode_label = "Major（長調）" if mode == "major" else "Minor（短調）"
    return f"{pitch_name(tonic, notation)} {mode_label}"


def camelot_code(key_index: int) -> str:
    tonic, mode = KEY_LABELS[int(key_index)]
    return MAJOR_CAMELOT[tonic] if mode == "major" else MINOR_CAMELOT[tonic]


def transpose_key(key_index: int, semitones: int) -> int:
    tonic, mode = KEY_LABELS[int(key_index)]
    shifted_tonic = (tonic + int(semitones)) % 12
    return shifted_tonic if mode == "major" else 12 + shifted_tonic


def relative_key_index(key_index: int) -> int:
    tonic, mode = KEY_LABELS[int(key_index)]
    if mode == "major":
        return 12 + ((tonic + 9) % 12)
    return (tonic + 3) % 12


def harmonic_family(key_index: int) -> int:
    tonic, mode = KEY_LABELS[int(key_index)]
    return tonic if mode == "major" else (tonic + 3) % 12


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes}:{remaining:04.1f}"


def format_shift(semitones: int) -> str:
    semitones = int(semitones) % 12

    if semitones == 0:
        return "0半音"

    signed = semitones if semitones <= 6 else semitones - 12
    return f"{signed:+d}半音"


# libsndfileが直接読める（ffmpeg不要の）形式。
NATIVE_AUDIO_FORMATS = {"wav", "flac", "ogg", "aiff", "aif"}


def _decode_error(suffix: str) -> ValueError:
    """
    音声をデコードできなかったときの、原因の分かるエラーを作る。

    m4a/AACやmp3はlibsndfileだけでは読めず、ffmpegが要る。
    Streamlit Cloudでは packages.txt がリポジトリ直下に無いと
    ffmpegが入らないため、NoBackendErrorになりやすい。
    そのままでは空メッセージの例外になり原因が分からないので、
    形式名と対処をここで添える。
    """
    if suffix in NATIVE_AUDIO_FORMATS:
        return ValueError(
            f"{suffix} ファイルを読み込めませんでした。"
            "ファイルが壊れていないか確認してください。"
        )

    return ValueError(
        f"{suffix} 形式の読み込みに失敗しました。"
        "この形式（m4a / mp3 など）のデコードには ffmpeg が必要です。\n\n"
        "・WAV / FLAC に変換してからアップロードすると確実です。\n"
        "・自分で運用している場合は、リポジトリ直下に "
        "`packages.txt`（内容は ffmpeg）を置き、アプリを再起動（Reboot）"
        "してください。"
    )


def format_bpm(tempo: dict[str, float]) -> str:
    """テンポ推定を人間向けの文字列にする。倍/半の候補も併記する。"""
    bpm = float(tempo.get("bpm", 0.0))

    if bpm <= 0.0:
        return "拍が弱く推定できませんでした"

    text = f"{bpm:.0f} BPM"

    alt = float(tempo.get("alt_bpm", 0.0))
    if alt > 0.0:
        text += f"（倍/半の候補: {alt:.0f} BPM）"

    return text


def bpm_confidence_label(confidence: float) -> str:
    if confidence >= 0.55:
        return "高め"
    if confidence >= 0.30:
        return "中程度"
    return "低め（倍/半の候補も検討してください）"


def safe_progress(
    callback: Callable | None,
    value: float,
    description: str,
) -> None:
    if callback is not None:
        callback(float(np.clip(value, 0.0, 1.0)), desc=description)


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    vector_a = np.asarray(vector_a, dtype=np.float64)
    vector_b = np.asarray(vector_b, dtype=np.float64)

    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator < 1e-12:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denominator)


def profile_scores(
    chroma_vector: np.ndarray,
    templates: np.ndarray,
) -> np.ndarray:
    chroma_vector = np.asarray(chroma_vector, dtype=np.float64)

    if chroma_vector.shape != (12,) or chroma_vector.std() < 1e-10:
        raise ValueError("音程成分を十分に取り出せませんでした。")

    normalized = _standardize(chroma_vector)
    return (templates @ normalized) / 12.0


def ensemble_key_scores(
    full_chroma: np.ndarray,
    bass_chroma: np.ndarray,
) -> np.ndarray:
    """
    全音域のプロファイル一致を主役にし、曖昧な場合だけ低音域の情報を強める。

    プロファイルはポップス寄りのShaathを主軸に、
    コーパス由来のAlbrecht、古典寄りのTemperley／Krumhanslを混ぜる。
    Krumhansl単体は探索実験由来で、ポップスでは属調・相対調へ寄りやすい。
    """
    krumhansl_scores = profile_scores(full_chroma, KRUMHANSL_TEMPLATES)
    temperley_scores = profile_scores(full_chroma, TEMPERLEY_TEMPLATES)
    shaath_scores = profile_scores(full_chroma, SHAATH_TEMPLATES)
    albrecht_scores = profile_scores(full_chroma, ALBRECHT_TEMPLATES)

    base_scores = (
        0.36 * shaath_scores
        + 0.25 * albrecht_scores
        + 0.22 * temperley_scores
        + 0.17 * krumhansl_scores
    )

    ranking = np.argsort(base_scores)[::-1]
    base_margin = float(base_scores[ranking[0]] - base_scores[ranking[1]])

    if base_margin >= 0.10:
        bass_weight = 0.035
    elif base_margin >= 0.065:
        bass_weight = 0.070
    elif base_margin >= 0.035:
        bass_weight = 0.120
    else:
        bass_weight = 0.180

    bass_scores = profile_scores(bass_chroma, BASS_TEMPLATES)
    return base_scores + bass_weight * bass_scores


def scale_change_evidence(
    before_chroma: np.ndarray,
    after_chroma: np.ndarray,
    before_key: int,
    shift: int,
) -> float:
    """
    「音階の構成音そのものが入れ替わったか」を測る。

    コード進行の重心がⅠからⅣ・Ⅴへ移っただけの区間は、
    クロマ全体の相関では±5半音の移調に見えてしまう。
    一方で本当に転調していれば、新しい調にしか無い音（C→Gの
    F♯など）が現れ、元の調にしか無い音（F）が減る。
    その増減だけを直接見ることで、進行の偏りと転調を切り分ける。
    """
    shift = int(shift) % 12

    if shift == 0:
        return 0.0

    family = harmonic_family(int(before_key))
    before_set = scale_membership(family)
    after_set = scale_membership(family + shift)

    gained = after_set & ~before_set
    lost = before_set & ~after_set

    if not gained.any() or not lost.any():
        return 0.0

    before_chroma = np.asarray(before_chroma, dtype=np.float64)
    after_chroma = np.asarray(after_chroma, dtype=np.float64)

    gained_delta = float(
        np.mean(after_chroma[gained] - before_chroma[gained])
    )
    lost_delta = float(
        np.mean(before_chroma[lost] - after_chroma[lost])
    )

    return gained_delta + lost_delta


def scale_change_profile(
    before_chroma: np.ndarray,
    after_chroma: np.ndarray,
    before_key: int,
) -> np.ndarray:
    return np.array(
        [
            scale_change_evidence(
                before_chroma,
                after_chroma,
                before_key,
                shift,
            )
            for shift in range(12)
        ],
        dtype=np.float64,
    )


def key_confidence(scores: np.ndarray) -> float:
    ranking = np.argsort(scores)[::-1]
    top = float(scores[ranking[0]])
    second = float(scores[ranking[1]])
    median = float(np.median(scores))

    margin_component = np.clip((top - second) / 0.15, 0.0, 1.0)
    tonality_component = np.clip((top - median) / 0.72, 0.0, 1.0)

    return float(100.0 * (0.68 * margin_component + 0.32 * tonality_component))


def confidence_label(value: float) -> str:
    if value >= 75:
        return "高め"
    if value >= 50:
        return "中程度"
    return "低め"


def weighted_feature_vector(
    feature: np.ndarray,
    rms: np.ndarray,
    frame_start: int,
    frame_end: int,
) -> np.ndarray:
    frame_end = min(frame_end, feature.shape[1], rms.shape[0])
    frame_start = max(0, min(frame_start, frame_end - 1))

    local_feature = feature[:, frame_start:frame_end]
    local_rms = rms[frame_start:frame_end]

    if local_feature.shape[1] < 2:
        raise ValueError("解析可能なフレームが不足しています。")

    floor = float(np.percentile(local_rms, 20))
    weights = np.maximum(local_rms, floor) + 1e-8

    vector = np.average(local_feature, axis=1, weights=weights)
    vector = np.maximum(vector, 0.0)
    vector /= vector.sum() + 1e-12

    return vector


def seconds_to_frame(seconds: float, sr: int) -> int:
    return int(
        librosa.time_to_frames(
            max(0.0, float(seconds)),
            sr=sr,
            hop_length=HOP_LENGTH,
        )
    )


def region_vectors(
    full_chroma: np.ndarray,
    bass_chroma: np.ndarray,
    rms: np.ndarray,
    sr: int,
    start_seconds: float,
    end_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    start_frame = seconds_to_frame(start_seconds, sr)
    end_frame = seconds_to_frame(end_seconds, sr)

    return (
        weighted_feature_vector(
            full_chroma,
            rms,
            start_frame,
            end_frame,
        ),
        weighted_feature_vector(
            bass_chroma,
            rms,
            start_frame,
            end_frame,
        ),
    )


def transposition_similarity(
    before_full: np.ndarray,
    after_full: np.ndarray,
    before_bass: np.ndarray,
    after_bass: np.ndarray,
) -> dict[str, Any]:
    full_similarities = np.array(
        [
            cosine_similarity(before_full, np.roll(after_full, -shift))
            for shift in range(12)
        ],
        dtype=np.float64,
    )
    bass_similarities = np.array(
        [
            cosine_similarity(before_bass, np.roll(after_bass, -shift))
            for shift in range(12)
        ],
        dtype=np.float64,
    )

    combined = 0.80 * full_similarities + 0.20 * bass_similarities

    best_shift = int(np.argmax(combined))
    nonzero_ranking = np.argsort(combined[1:])[::-1] + 1
    best_nonzero_shift = int(nonzero_ranking[0])

    same_similarity = float(combined[0])
    best_nonzero_similarity = float(combined[best_nonzero_shift])
    gain = best_nonzero_similarity - same_similarity
    change_distance = 1.0 - same_similarity

    strength = (
        0.52 * np.clip((gain - 0.010) / 0.150, 0.0, 1.0)
        + 0.30 * np.clip((change_distance - 0.045) / 0.300, 0.0, 1.0)
        + 0.18 * np.clip(
            (best_nonzero_similarity - 0.55) / 0.35,
            0.0,
            1.0,
        )
    )

    return {
        "best_shift": best_shift,
        "best_nonzero_shift": best_nonzero_shift,
        "same_similarity": same_similarity,
        "best_nonzero_similarity": best_nonzero_similarity,
        "gain": float(gain),
        "distance": float(change_distance),
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "combined_similarities": combined,
    }



def softmax_key_distribution(
    scores: np.ndarray,
    temperature: float = 0.085,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    centered = (scores - np.max(scores)) / max(temperature, 1e-6)
    centered = np.clip(centered, -50.0, 0.0)
    probabilities = np.exp(centered)
    return probabilities / (probabilities.sum() + 1e-12)


def shift_key_distribution(
    probabilities: np.ndarray,
    semitones: int,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    semitones = int(semitones) % 12

    return np.concatenate(
        [
            np.roll(probabilities[:12], semitones),
            np.roll(probabilities[12:], semitones),
        ]
    )


def hybrid_shift_analysis(
    before_full: np.ndarray,
    after_full: np.ndarray,
    before_bass: np.ndarray,
    after_bass: np.ndarray,
) -> dict[str, Any]:
    """
    音の分布そのものと、24キー候補の分布を両方使って
    転調量を推定する。

    相対長短調は24キー分布上で同時に移動するため、
    G major / E minor の曖昧さがあっても
    +1半音などの移動方向を保ちやすい。
    """
    chroma_result = transposition_similarity(
        before_full,
        after_full,
        before_bass,
        after_bass,
    )

    before_scores = ensemble_key_scores(before_full, before_bass)
    after_scores = ensemble_key_scores(after_full, after_bass)

    before_probabilities = softmax_key_distribution(before_scores)
    after_probabilities = softmax_key_distribution(after_scores)

    key_alignment = np.array(
        [
            cosine_similarity(
                shift_key_distribution(before_probabilities, shift),
                after_probabilities,
            )
            for shift in range(12)
        ],
        dtype=np.float64,
    )

    chroma_alignment = np.asarray(
        chroma_result["combined_similarities"],
        dtype=np.float64,
    )

    before_winner = int(np.argmax(before_scores))
    after_winner = int(np.argmax(after_scores))

    scale_change = scale_change_profile(
        before_full,
        after_full,
        before_winner,
    )
    scale_support = np.clip(scale_change / SCALE_CHANGE_SCALE, -1.0, 1.0)

    combined_alignment = (
        0.52 * chroma_alignment
        + 0.38 * key_alignment
        + 0.10 * (0.5 + 0.5 * scale_support)
    )

    nonzero_ranking = np.argsort(combined_alignment[1:])[::-1] + 1
    best_nonzero_shift = int(nonzero_ranking[0])

    same_alignment = float(combined_alignment[0])
    best_nonzero_alignment = float(
        combined_alignment[best_nonzero_shift]
    )
    hybrid_gain = best_nonzero_alignment - same_alignment

    direct_shift = (
        KEY_LABELS[after_winner][0]
        - KEY_LABELS[before_winner][0]
    ) % 12

    separate_fit = 0.5 * (
        float(np.max(before_scores))
        + float(np.max(after_scores))
    )
    same_key_fit = float(
        np.max(0.5 * (before_scores + after_scores))
    )
    segmentation_gain = max(0.0, separate_fit - same_key_fit)

    key_distribution_change = (
        1.0
        - cosine_similarity(
            before_probabilities,
            after_probabilities,
        )
    )

    chroma_distance = float(chroma_result["distance"])

    structural_strength = float(
        np.clip(
            0.38 * np.clip(chroma_distance / 0.30, 0.0, 1.0)
            + 0.37 * np.clip(
                key_distribution_change / 0.42,
                0.0,
                1.0,
            )
            + 0.25 * np.clip(
                segmentation_gain / 0.11,
                0.0,
                1.0,
            ),
            0.0,
            1.0,
        )
    )

    best_scale_support = float(
        np.clip(scale_support[best_nonzero_shift], -1.0, 1.0)
    )

    shift_strength = float(
        np.clip(
            0.36 * np.clip(
                (hybrid_gain - 0.002) / 0.105,
                0.0,
                1.0,
            )
            + 0.18 * np.clip(
                (best_nonzero_alignment - 0.54) / 0.38,
                0.0,
                1.0,
            )
            + 0.22 * structural_strength
            + 0.24 * np.clip(best_scale_support, 0.0, 1.0),
            0.0,
            1.0,
        )
    )

    return {
        **chroma_result,
        "scale_change": scale_change,
        "scale_support": scale_support,
        "scale_change_evidence": float(
            scale_change[best_nonzero_shift]
        ),
        "before_scores": before_scores,
        "after_scores": after_scores,
        "before_winner": before_winner,
        "after_winner": after_winner,
        "direct_shift": int(direct_shift),
        "key_alignment": key_alignment,
        "combined_alignment": combined_alignment,
        "best_nonzero_shift": best_nonzero_shift,
        "same_alignment": same_alignment,
        "best_nonzero_alignment": best_nonzero_alignment,
        "hybrid_gain": float(hybrid_gain),
        "segmentation_gain": float(segmentation_gain),
        "key_distribution_change": float(key_distribution_change),
        "structural_strength": structural_strength,
        "shift_strength": shift_strength,
    }


def combine_multiscale_analyses(
    analyses: list[dict[str, Any]],
) -> dict[str, Any]:
    if not analyses:
        raise ValueError("転調候補を評価できませんでした。")

    shift_votes = np.zeros(12, dtype=np.float64)

    for analysis in analyses:
        shift = int(analysis["best_nonzero_shift"])
        vote_weight = (
            0.52 * analysis["shift_strength"]
            + 0.28 * analysis["structural_strength"]
            + 0.20 * np.clip(
                analysis["best_nonzero_alignment"],
                0.0,
                1.0,
            )
        )
        shift_votes[shift] += max(float(vote_weight), 1e-6)

        # 区間単独判定の主音差が一致した場合は補助票を加える
        direct_shift = int(analysis["direct_shift"])
        if direct_shift != 0:
            shift_votes[direct_shift] += (
                0.12
                * analysis["structural_strength"]
            )

    chosen_shift = int(np.argmax(shift_votes[1:]) + 1)
    total_votes = float(np.sum(shift_votes[1:]))
    consensus = (
        float(shift_votes[chosen_shift] / total_votes)
        if total_votes > 1e-12
        else 0.0
    )

    matching = [
        analysis
        for analysis in analyses
        if int(analysis["best_nonzero_shift"]) == chosen_shift
    ]
    if not matching:
        matching = analyses

    weights = np.array(
        [
            max(
                0.05,
                analysis["shift_strength"]
                + 0.35 * analysis["structural_strength"],
            )
            for analysis in matching
        ],
        dtype=np.float64,
    )

    def weighted_average(field: str) -> float:
        values = np.array(
            [float(analysis[field]) for analysis in matching],
            dtype=np.float64,
        )
        return float(np.average(values, weights=weights))

    strongest = max(
        analyses,
        key=lambda analysis: (
            analysis["shift_strength"],
            analysis["structural_strength"],
        ),
    )

    candidate_strength = float(
        np.clip(
            0.46 * max(
                analysis["shift_strength"]
                for analysis in analyses
            )
            + 0.24 * weighted_average("shift_strength")
            + 0.18 * weighted_average("structural_strength")
            + 0.12 * consensus,
            0.0,
            1.0,
        )
    )

    combined_alignment = np.average(
        np.stack(
            [
                analysis["combined_alignment"]
                for analysis in matching
            ]
        ),
        axis=0,
        weights=weights,
    )

    # 構成音の入れ替わりは時間幅を変えても一貫して出るはずなので、
    # 全スケールの中で最も弱い値を採用して過検出を抑える。
    scale_change_stack = np.stack(
        [
            np.asarray(analysis["scale_change"], dtype=np.float64)
            for analysis in analyses
        ]
    )
    scale_change = np.min(scale_change_stack, axis=0)
    scale_support = np.clip(
        scale_change / SCALE_CHANGE_SCALE,
        -1.0,
        1.0,
    )

    same_alignment = float(combined_alignment[0])
    best_nonzero_alignment = float(
        combined_alignment[chosen_shift]
    )

    return {
        **strongest,
        "shift": chosen_shift,
        "best_nonzero_shift": chosen_shift,
        "combined_alignment": combined_alignment,
        "scale_change": scale_change,
        "scale_support": scale_support,
        "scale_change_evidence": float(scale_change[chosen_shift]),
        "same_alignment": same_alignment,
        "best_nonzero_alignment": best_nonzero_alignment,
        "hybrid_gain": (
            best_nonzero_alignment - same_alignment
        ),
        "gain": (
            best_nonzero_alignment - same_alignment
        ),
        "strength": candidate_strength,
        "candidate_strength": candidate_strength,
        "structural_strength": weighted_average(
            "structural_strength"
        ),
        "segmentation_gain": weighted_average(
            "segmentation_gain"
        ),
        "consensus": consensus,
        "scale_count": len(analyses),
        "matching_scale_count": len(matching),
    }


def scan_change_candidates(
    full_chroma: np.ndarray,
    bass_chroma: np.ndarray,
    rms: np.ndarray,
    sr: int,
    duration: float,
    settings: dict[str, float],
) -> list[dict[str, Any]]:
    minimum_segment = float(settings["minimum_segment_seconds"])
    maximum_context = max(CHANGE_CONTEXT_OPTIONS)

    first_time = max(minimum_segment, min(CHANGE_CONTEXT_OPTIONS))
    last_time = duration - minimum_segment

    if last_time <= first_time:
        return []

    candidates: list[dict[str, Any]] = []

    for time_seconds in np.arange(
        first_time,
        last_time + 1e-9,
        CHANGE_SCAN_STEP_SECONDS,
    ):
        scale_analyses: list[dict[str, Any]] = []

        for context_seconds in CHANGE_CONTEXT_OPTIONS:
            before_start = time_seconds - context_seconds
            before_end = time_seconds - CHANGE_GUARD_SECONDS
            after_start = time_seconds + CHANGE_GUARD_SECONDS
            after_end = time_seconds + context_seconds

            if before_start < 0.0 or after_end > duration:
                continue

            before_full, before_bass = region_vectors(
                full_chroma,
                bass_chroma,
                rms,
                sr,
                before_start,
                before_end,
            )
            after_full, after_bass = region_vectors(
                full_chroma,
                bass_chroma,
                rms,
                sr,
                after_start,
                after_end,
            )

            analysis = hybrid_shift_analysis(
                before_full,
                after_full,
                before_bass,
                after_bass,
            )
            analysis["context_seconds"] = float(context_seconds)
            scale_analyses.append(analysis)

        if not scale_analyses:
            continue

        combined = combine_multiscale_analyses(scale_analyses)

        # v0.4では候補を少し広めに残し、後段の状態追跡で
        # 「本当にキーが変わったか」を全区間まとめて判断する。
        primary_pass = (
            combined["candidate_strength"]
            >= settings["candidate_threshold"] * 0.80
            and combined["hybrid_gain"]
            >= settings["minimum_gain"] * 0.55
            and combined["structural_strength"]
            >= settings["minimum_structure"] * 0.82
            and combined["best_nonzero_alignment"] >= 0.50
        )

        structural_fallback = (
            combined["candidate_strength"]
            >= settings["candidate_threshold"] * 0.70
            and combined["structural_strength"]
            >= settings["minimum_structure"] * 0.88
            and combined["segmentation_gain"] >= 0.011
            and combined["consensus"] >= 0.28
            and combined["best_nonzero_alignment"] >= 0.48
        )

        # 構成音が実際に入れ替わっていない候補は、コード進行の重心移動と
        # 見なして早い段階で落とす。5度方向はとくに紛れやすいので厳しく。
        scale_evidence = float(combined["scale_change_evidence"])
        scale_floor = float(settings["minimum_scale_change"])
        chosen_shift = int(combined["shift"]) % 12

        if chosen_shift in FIFTH_SHIFTS:
            scale_gate = scale_evidence >= scale_floor
        else:
            scale_gate = scale_evidence >= scale_floor * 0.35

        if scale_gate and (primary_pass or structural_fallback):
            candidates.append(
                {
                    "time": float(time_seconds),
                    **combined,
                    "detection_route": (
                        "移調量"
                        if primary_pass
                        else "調性変化フォールバック"
                    ),
                }
            )

    return candidates


def select_candidate_peaks(
    candidates: list[dict[str, Any]],
    settings: dict[str, float],
) -> list[dict[str, Any]]:
    """
    同じ転調点の周囲に並ぶ候補を時間クラスタへまとめ、
    各クラスタから最も強い候補を1件残す。

    強い候補だけを全曲から上位順に取る方式ではなく、
    時系列に沿って局所ピークを残すため、転調後の復帰や
    3回以上の転調も候補から落ちにくい。
    """
    if not candidates:
        return []

    cluster_radius = max(
        4.0,
        float(settings["minimum_boundary_gap"]) * 0.45,
    )

    ordered_by_time = sorted(candidates, key=lambda item: item["time"])
    clusters: list[list[dict[str, Any]]] = []
    current_cluster: list[dict[str, Any]] = []

    for candidate in ordered_by_time:
        if (
            current_cluster
            and candidate["time"] - current_cluster[-1]["time"]
            > cluster_radius
        ):
            clusters.append(current_cluster)
            current_cluster = []
        current_cluster.append(candidate)

    if current_cluster:
        clusters.append(current_cluster)

    local_peaks = [
        max(
            cluster,
            key=lambda candidate: (
                candidate["candidate_strength"],
                candidate["structural_strength"],
                candidate["consensus"],
            ),
        )
        for cluster in clusters
    ]

    minimum_gap = float(settings["minimum_boundary_gap"])
    selected: list[dict[str, Any]] = []

    for candidate in sorted(
        local_peaks,
        key=lambda item: (
            item["candidate_strength"],
            item["structural_strength"],
        ),
        reverse=True,
    ):
        conflicts = [
            existing
            for existing in selected
            if abs(candidate["time"] - existing["time"]) < minimum_gap
        ]

        if not conflicts:
            selected.append(candidate)
            continue

        # 近接する2候補でも、どちらも十分強く、その間が短い転調区間として
        # 成立しうる場合は残す。状態追跡が不要な方を後で消す。
        if (
            candidate["candidate_strength"] >= 0.62
            and candidate["structural_strength"] >= 0.34
            and all(
                existing["candidate_strength"] >= 0.62
                and existing["structural_strength"] >= 0.34
                for existing in conflicts
            )
        ):
            selected.append(candidate)

        if len(selected) >= MAX_BOUNDARIES:
            break

    return sorted(selected[:MAX_BOUNDARIES], key=lambda item: item["time"])


def refine_boundary(
    candidate: dict[str, Any],
    full_chroma: np.ndarray,
    bass_chroma: np.ndarray,
    rms: np.ndarray,
    sr: int,
    duration: float,
) -> dict[str, Any]:
    original_time = float(candidate["time"])
    expected_shift = int(candidate["shift"])

    search_start = max(
        min(CHANGE_CONTEXT_OPTIONS),
        original_time - BOUNDARY_REFINEMENT_RADIUS,
    )
    search_end = min(
        duration - min(CHANGE_CONTEXT_OPTIONS),
        original_time + BOUNDARY_REFINEMENT_RADIUS,
    )

    best: dict[str, Any] | None = None

    for time_seconds in np.arange(search_start, search_end + 1e-9, 1.0):
        scale_analyses: list[dict[str, Any]] = []

        for context_seconds in CHANGE_CONTEXT_OPTIONS:
            if (
                time_seconds - context_seconds < 0.0
                or time_seconds + context_seconds > duration
            ):
                continue

            before_full, before_bass = region_vectors(
                full_chroma,
                bass_chroma,
                rms,
                sr,
                time_seconds - context_seconds,
                time_seconds - 1.0,
            )
            after_full, after_bass = region_vectors(
                full_chroma,
                bass_chroma,
                rms,
                sr,
                time_seconds + 1.0,
                time_seconds + context_seconds,
            )

            analysis = hybrid_shift_analysis(
                before_full,
                after_full,
                before_bass,
                after_bass,
            )
            analysis["context_seconds"] = float(context_seconds)
            scale_analyses.append(analysis)

        if not scale_analyses:
            continue

        combined = combine_multiscale_analyses(scale_analyses)
        shift_alignment = float(
            combined["combined_alignment"][expected_shift]
        )

        local_score = float(
            0.38 * combined["candidate_strength"]
            + 0.27 * combined["structural_strength"]
            + 0.20 * np.clip(
                shift_alignment - combined["same_alignment"],
                0.0,
                1.0,
            )
            + 0.15 * combined["consensus"]
        )

        record = {
            **combined,
            "time": float(time_seconds),
            "shift": expected_shift,
            "best_nonzero_shift": expected_shift,
            "best_nonzero_alignment": shift_alignment,
            "gain": (
                shift_alignment - combined["same_alignment"]
            ),
            "hybrid_gain": (
                shift_alignment - combined["same_alignment"]
            ),
            "local_score": local_score,
            "detection_route": candidate.get(
                "detection_route",
                "ハイブリッド",
            ),
        }

        if best is None or record["local_score"] > best["local_score"]:
            best = record

    return best if best is not None else candidate


def deduplicate_refined_boundaries(
    boundaries: list[dict[str, Any]],
    minimum_gap: float,
) -> list[dict[str, Any]]:
    if not boundaries:
        return []

    ordered = sorted(
        boundaries,
        key=lambda boundary: boundary["strength"],
        reverse=True,
    )
    kept: list[dict[str, Any]] = []

    for boundary in ordered:
        if all(
            abs(boundary["time"] - existing["time"]) >= minimum_gap
            for existing in kept
        ):
            kept.append(boundary)

    return sorted(kept, key=lambda boundary: boundary["time"])



def shift_aware_boundary_pass(
    boundary: dict[str, Any],
    settings: dict[str, float],
) -> bool:
    """
    半音移動は比較的拾いやすくし、
    5度方向の移動はコード進行との混同を避けるため厳しく判定する。
    """
    shift = int(boundary.get("shift", 0)) % 12
    alignment = float(
        boundary.get(
            "best_nonzero_alignment",
            boundary.get("best_nonzero_similarity", 0.0),
        )
    )
    gain = float(
        boundary.get(
            "hybrid_gain",
            boundary.get("gain", 0.0),
        )
    )
    structural = float(
        boundary.get("structural_strength", 0.0)
    )
    segmentation_gain = float(
        boundary.get("segmentation_gain", 0.0)
    )
    consensus = float(boundary.get("consensus", 0.0))
    matching_scales = int(
        boundary.get("matching_scale_count", 1)
    )
    quality = boundary_confidence(boundary)
    scale_evidence = float(
        boundary.get("scale_change_evidence", 0.0)
    )
    scale_floor = float(settings["minimum_scale_change"])

    base_primary = (
        gain >= settings["minimum_gain"] * 0.65
        and alignment >= 0.54
    )
    base_structural = (
        structural >= settings["minimum_structure"] * 1.05
        and segmentation_gain >= 0.016
        and consensus >= 0.38
    )

    if shift in FIFTH_SHIFTS:
        # ±5半音は属調・下属調、コード構成音の変化でも出やすい。
        # 調号が1つ動いたこと（F→F♯など）を必須条件にする。
        return bool(
            scale_evidence >= scale_floor * 1.6
            and quality >= 76.0
            and alignment >= 0.62
            and structural >= max(
                0.40,
                settings["minimum_structure"] * 1.35,
            )
            and segmentation_gain >= 0.032
            and consensus >= 0.58
            and matching_scales >= 2
            and (base_primary or base_structural)
        )

    if shift in SEMITONE_SHIFTS:
        # 短い終盤転調を落としにくくする
        return bool(
            scale_evidence >= scale_floor * 0.45
            and alignment >= 0.52
            and (
                gain >= settings["minimum_gain"] * 0.50
                or (
                    structural
                    >= settings["minimum_structure"] * 0.95
                    and segmentation_gain >= 0.013
                    and consensus >= 0.34
                )
            )
        )

    return bool(
        scale_evidence >= scale_floor * 0.70
        and (base_primary or base_structural)
    )


def state_runs(state_path: np.ndarray) -> list[tuple[int, int, int]]:
    """状態列を (開始index, 終了index, キー) の連続区間へまとめる。"""
    path = np.asarray(state_path, dtype=int)

    if path.size == 0:
        return []

    runs: list[tuple[int, int, int]] = []
    run_start = 0
    run_state = int(path[0])

    for index in range(1, len(path)):
        current_state = int(path[index])
        if current_state != run_state:
            runs.append((run_start, index - 1, run_state))
            run_start = index
            run_state = current_state

    runs.append((run_start, len(path) - 1, run_state))
    return runs


def suppress_fifth_round_trips(
    state_path: np.ndarray,
    segment_results: list[dict[str, Any]],
    candidate_boundaries: list[dict[str, Any]],
) -> np.ndarray:
    """
    -5半音の後に+5半音など、5度方向へ移動して元へ戻る往復を打ち消す。

    Ⅳ・Ⅴを強調するサビや二次ドミナントの多い進行は、区間単位で見ると
    属調・下属調へ移ったように見え、次の区間で必ず元へ戻る。
    根拠が非常に強い場合を除き、機能和声による疑似転調として吸収する。
    """
    path = np.asarray(state_path, dtype=int).copy()

    for _ in range(4):
        runs = state_runs(path)

        if len(runs) < 3:
            break

        collapsed = False

        for run_index in range(1, len(runs) - 1):
            previous_state = runs[run_index - 1][2]
            current_state = runs[run_index][2]
            next_state = runs[run_index + 1][2]

            if previous_state != next_state or current_state == previous_state:
                continue

            shift = (
                KEY_LABELS[current_state][0]
                - KEY_LABELS[previous_state][0]
            ) % 12

            if shift not in FIFTH_SHIFTS:
                continue

            enter_index = runs[run_index][0] - 1
            leave_index = runs[run_index + 1][0] - 1

            if not (
                0 <= enter_index < len(candidate_boundaries)
                and 0 <= leave_index < len(candidate_boundaries)
            ):
                continue

            first = candidate_boundaries[enter_index]
            second = candidate_boundaries[leave_index]

            middle = segment_results[
                runs[run_index][0]:runs[run_index][1] + 1
            ]
            middle_confidence = (
                min(
                    float(segment["absolute_confidence"])
                    for segment in middle
                )
                if middle
                else 0.0
            )
            middle_duration = sum(
                float(segment["duration"]) for segment in middle
            )

            qualities = (
                boundary_confidence(first),
                boundary_confidence(second),
            )
            consensuses = (
                float(first.get("consensus", 0.0)),
                float(second.get("consensus", 0.0)),
            )
            structures = (
                float(first.get("structural_strength", 0.0)),
                float(second.get("structural_strength", 0.0)),
            )
            scale_evidences = (
                float(first.get("scale_change_evidence", 0.0)),
                float(second.get("scale_change_evidence", 0.0)),
            )

            # 本当に「転調して元へ戻った」とみなすには、両境界と
            # 中間区間のすべてにかなり強い証拠を要求する。
            exceptionally_strong = (
                min(qualities) >= 88.0
                and min(consensuses) >= 0.70
                and min(structures) >= 0.50
                and min(scale_evidences) >= 0.045
                and middle_confidence >= 68.0
                and middle_duration >= 24.0
            )

            if not exceptionally_strong:
                path[runs[run_index][0]:runs[run_index][1] + 1] = (
                    previous_state
                )
                collapsed = True
                break

        if not collapsed:
            break

    return path




def make_segment_ranges(
    duration: float,
    boundaries: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    points = [0.0]
    points.extend(float(boundary["time"]) for boundary in boundaries)
    points.append(float(duration))

    return [
        (points[index], points[index + 1])
        for index in range(len(points) - 1)
    ]


def segment_key_analysis(
    full_chroma: np.ndarray,
    bass_chroma: np.ndarray,
    rms: np.ndarray,
    sr: int,
    start_seconds: float,
    end_seconds: float,
) -> dict[str, Any]:
    segment_length = end_seconds - start_seconds
    margin = min(4.0, max(0.0, segment_length * 0.08))

    analysis_start = start_seconds + margin
    analysis_end = end_seconds - margin

    if analysis_end - analysis_start < 8.0:
        analysis_start = start_seconds
        analysis_end = end_seconds

    full_vector, bass_vector = region_vectors(
        full_chroma,
        bass_chroma,
        rms,
        sr,
        analysis_start,
        analysis_end,
    )
    scores = ensemble_key_scores(full_vector, bass_vector)
    ranking = np.argsort(scores)[::-1]
    winner = int(ranking[0])

    return {
        "start": float(start_seconds),
        "end": float(end_seconds),
        "duration": float(segment_length),
        "full_vector": full_vector,
        "bass_vector": bass_vector,
        "scores": scores,
        "absolute_key": winner,
        "absolute_second": int(ranking[1]),
        "absolute_confidence": key_confidence(scores),
    }


def _mode_relation_score(previous_key: int, current_key: int, shift: int) -> float:
    same_mode_key = transpose_key(previous_key, shift)
    relative_mode_key = relative_key_index(same_mode_key)

    if current_key == same_mode_key:
        return 0.20
    if current_key == relative_mode_key:
        return 0.06
    return -0.18


def optimize_segment_key_sequence(
    segment_results: list[dict[str, Any]],
    candidate_boundaries: list[dict[str, Any]],
    settings: dict[str, float],
) -> np.ndarray:
    """
    全候補区間を24調の状態列としてまとめて最適化する。

    境界ごとに独立して前区間からキーを足し算するのではなく、
    各区間のキーらしさと境界の移調根拠を同時に評価するため、
    A→B→Aの復帰やA→B→C→Dの連続転調を同じ仕組みで扱える。
    """
    if not segment_results:
        return np.array([], dtype=int)

    n_segments = len(segment_results)
    n_states = 24
    emissions = np.zeros((n_segments, n_states), dtype=np.float64)

    for index, segment in enumerate(segment_results):
        probabilities = softmax_key_distribution(
            segment["scores"],
            temperature=0.105,
        )
        log_probabilities = np.log(probabilities + 1e-12)
        log_probabilities -= np.max(log_probabilities)

        duration_weight = float(
            np.clip(
                math.sqrt(max(segment["duration"], 1.0) / 18.0),
                0.72,
                1.85,
            )
        )
        confidence_weight = 0.78 + 0.34 * np.clip(
            segment["absolute_confidence"] / 100.0,
            0.0,
            1.0,
        )
        emissions[index] = (
            log_probabilities
            * duration_weight
            * confidence_weight
            * 0.58
        )

    dp = np.full((n_segments, n_states), -np.inf, dtype=np.float64)
    back = np.zeros((n_segments, n_states), dtype=np.int16)
    dp[0] = emissions[0]

    base_penalty = float(settings.get("transition_penalty", 0.84))

    for segment_index in range(1, n_segments):
        boundary = candidate_boundaries[segment_index - 1]
        quality = boundary_confidence(boundary) / 100.0
        structural = float(boundary.get("structural_strength", 0.0))
        consensus = float(boundary.get("consensus", 0.0))
        expected_shift = int(boundary.get("shift", 0)) % 12
        alignments = np.asarray(
            boundary.get("combined_alignment", np.zeros(12)),
            dtype=np.float64,
        )
        scale_support = np.asarray(
            boundary.get("scale_support", np.zeros(12)),
            dtype=np.float64,
        )
        if scale_support.size != 12:
            scale_support = np.zeros(12, dtype=np.float64)

        transition_matrix = np.full(
            (n_states, n_states),
            -base_penalty,
            dtype=np.float64,
        )

        for previous_key in range(n_states):
            previous_tonic, _ = KEY_LABELS[previous_key]

            for current_key in range(n_states):
                if current_key == previous_key:
                    # キー維持を基準点(0)に置く。候補が弱いほど維持を優遇し、
                    # 強い候補でも維持そのものを不利にはしない。
                    # 転調側は下の加点で基準点を超えたときだけ選ばれる。
                    transition_matrix[previous_key, current_key] = (
                        0.30 * (1.0 - quality)
                    )
                    continue

                current_tonic, _ = KEY_LABELS[current_key]
                shift = (current_tonic - previous_tonic) % 12
                alignment = float(alignments[shift]) if alignments.size == 12 else 0.0

                alignment_support = np.clip(
                    (alignment - 0.45) / 0.45,
                    0.0,
                    1.0,
                )
                expected_bonus = 0.45 if shift == expected_shift else 0.0
                mode_score = _mode_relation_score(
                    previous_key,
                    current_key,
                    shift,
                )

                # 調号が実際に動いた証拠。負なら「構成音は変わっていない」
                # ということなので、そのまま減点として効く。
                scale_score = float(np.clip(scale_support[shift], -1.0, 1.0))

                fifth_penalty = 0.0
                if shift in FIFTH_SHIFTS:
                    # 属調・下属調は進行の重心移動でも出る。
                    # 構成音の入れ替わりが弱いほど強く抑える。
                    fifth_penalty = (
                        0.60 * (1.0 - np.clip(scale_score, 0.0, 1.0))
                        + 0.20 * (1.0 - quality)
                    )

                transition_matrix[previous_key, current_key] = (
                    -base_penalty
                    + 1.15 * quality * alignment_support
                    + expected_bonus * quality
                    + 0.35 * structural * quality
                    + 0.15 * consensus
                    + 0.55 * scale_score
                    + mode_score
                    - fifth_penalty
                )

        scores_from_previous = dp[segment_index - 1][:, None] + transition_matrix
        best_previous = np.argmax(scores_from_previous, axis=0)
        dp[segment_index] = (
            scores_from_previous[best_previous, np.arange(n_states)]
            + emissions[segment_index]
        )
        back[segment_index] = best_previous.astype(np.int16)

    path = np.zeros(n_segments, dtype=np.int16)
    path[-1] = int(np.argmax(dp[-1]))

    for segment_index in range(n_segments - 1, 0, -1):
        path[segment_index - 1] = back[segment_index, path[segment_index]]

    return path.astype(int)


def stabilize_short_state_excursions(
    state_path: np.ndarray,
    segment_results: list[dict[str, Any]],
    candidate_boundaries: list[dict[str, Any]],
    settings: dict[str, float],
) -> np.ndarray:
    """非常に短く根拠の弱いA→B→Aだけを吸収する。"""
    path = np.asarray(state_path, dtype=int).copy()

    if len(path) < 3:
        return path

    threshold = float(settings["minimum_segment_seconds"]) * 0.72

    for index in range(1, len(path) - 1):
        if path[index - 1] != path[index + 1] or path[index] == path[index - 1]:
            continue

        segment = segment_results[index]
        left = candidate_boundaries[index - 1]
        right = candidate_boundaries[index]
        left_quality = boundary_confidence(left)
        right_quality = boundary_confidence(right)
        weakest_scale = min(
            float(left.get("scale_change_evidence", 0.0)),
            float(right.get("scale_change_evidence", 0.0)),
        )

        weak_excursion = (
            segment["duration"] < threshold
            and (
                segment["absolute_confidence"] < 62.0
                or weakest_scale < float(settings["minimum_scale_change"])
            )
            and min(left_quality, right_quality) < 70.0
        )

        if weak_excursion:
            path[index] = path[index - 1]

    return path


def merge_optimized_state_runs(
    segment_results: list[dict[str, Any]],
    candidate_boundaries: list[dict[str, Any]],
    state_path: np.ndarray,
    full_chroma: np.ndarray,
    bass_chroma: np.ndarray,
    rms: np.ndarray,
    sr: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    同じ状態が続く候補区間を結合する。
    状態が変わった境界だけを採用し、それ以外は疑似転調候補として残す。
    """
    if not segment_results:
        return [], [], []

    runs = state_runs(state_path)

    merged_segments: list[dict[str, Any]] = []
    accepted_boundaries: list[dict[str, Any]] = []
    suppressed_boundaries: list[dict[str, Any]] = []

    accepted_cut_indices = {end for _, end, _ in runs[:-1]}

    for boundary_index, boundary in enumerate(candidate_boundaries):
        if boundary_index in accepted_cut_indices:
            updated = dict(boundary)
            previous_key = int(state_path[boundary_index])
            current_key = int(state_path[boundary_index + 1])
            previous_tonic, _ = KEY_LABELS[previous_key]
            current_tonic, _ = KEY_LABELS[current_key]
            updated["raw_shift"] = int(boundary.get("shift", 0))
            updated["shift"] = int((current_tonic - previous_tonic) % 12)
            updated["state_before"] = previous_key
            updated["state_after"] = current_key
            updated["detection_route"] = (
                str(boundary.get("detection_route", "ハイブリッド"))
                + "＋状態追跡"
            )
            accepted_boundaries.append(updated)
        else:
            updated = dict(boundary)
            updated["suppression_reason"] = "前後の最適キー状態が同じ"
            suppressed_boundaries.append(updated)

    seen_states: set[int] = set()

    for run_number, (start_index, end_index, state) in enumerate(runs):
        start_seconds = float(segment_results[start_index]["start"])
        end_seconds = float(segment_results[end_index]["end"])
        merged = segment_key_analysis(
            full_chroma,
            bass_chroma,
            rms,
            sr,
            start_seconds,
            end_seconds,
        )
        merged["final_key"] = int(state)

        if run_number == 0:
            method = "開始区間を状態追跡"
        elif state in seen_states:
            method = "過去のキーへ復帰"
        else:
            method = "複数転調の状態追跡"

        merged["method"] = method
        merged["state_run_start"] = start_index
        merged["state_run_end"] = end_index
        merged_segments.append(merged)
        seen_states.add(int(state))

    return merged_segments, accepted_boundaries, suppressed_boundaries


def aggregate_main_key(
    segments: list[dict[str, Any]],
) -> tuple[int, float, np.ndarray, float]:
    duration_by_key: dict[int, float] = {}

    for segment in segments:
        key_index = int(segment["final_key"])
        duration_by_key[key_index] = (
            duration_by_key.get(key_index, 0.0)
            + float(segment["duration"])
        )

    # 滞在時間が最長のキーを主調にする。ただし転調が曲のちょうど
    # 真ん中で起きた場合など僅差になることがあるので、その場合は
    # 曲が始まったキーをホームキーとみなす。
    start_key = int(segments[0]["final_key"])
    longest_duration = max(duration_by_key.values())

    def main_key_rank(key_index: int) -> tuple[bool, bool, float]:
        duration = duration_by_key[key_index]
        return (
            duration >= longest_duration * 0.85,
            key_index == start_key,
            duration,
        )

    main_key = max(duration_by_key, key=main_key_rank)
    total_duration = sum(duration_by_key.values())
    main_duration = duration_by_key[main_key]

    matching = [
        segment for segment in segments
        if int(segment["final_key"]) == main_key
    ]
    weights = np.array(
        [float(segment["duration"]) for segment in matching],
        dtype=np.float64,
    )
    main_scores = np.average(
        np.stack([segment["scores"] for segment in matching]),
        axis=0,
        weights=weights,
    )
    confidence = float(
        np.average(
            [key_confidence(segment["scores"]) for segment in matching],
            weights=weights,
        )
    )

    return (
        int(main_key),
        confidence,
        main_scores,
        float(main_duration / max(total_duration, 1e-9)),
    )


def overall_modulation_confidence(
    boundaries: list[dict[str, Any]],
) -> tuple[str, float]:
    if not boundaries:
        return "低い", 0.0

    strongest = max(
        boundary_confidence(boundary)
        for boundary in boundaries
    )

    strongest = float(np.clip(strongest, 0.0, 100.0))

    if strongest >= 70:
        return "高い", strongest
    if strongest >= 43:
        return "可能性あり", strongest
    return "低め", strongest


def boundary_confidence(boundary: dict[str, Any]) -> float:
    alignment = float(
        boundary.get(
            "best_nonzero_alignment",
            boundary.get("best_nonzero_similarity", 0.0),
        )
    )
    hybrid_gain = float(
        boundary.get(
            "hybrid_gain",
            boundary.get("gain", 0.0),
        )
    )
    structural_strength = float(
        boundary.get("structural_strength", 0.0)
    )
    consensus = float(boundary.get("consensus", 0.0))

    value = 100.0 * (
        0.36 * boundary.get("strength", 0.0)
        + 0.24 * np.clip(
            hybrid_gain / 0.11,
            0.0,
            1.0,
        )
        + 0.22 * structural_strength
        + 0.10 * consensus
        + 0.08 * np.clip(
            (alignment - 0.52) / 0.40,
            0.0,
            1.0,
        )
    )
    return float(np.clip(value, 0.0, 100.0))


def trim_silence(y: np.ndarray, top_db: float = 42.0) -> np.ndarray:
    """
    前後の無音を削る。

    librosa.effects.trimは信号を重ねてフレーム化するため、
    20分の曲では一時的に200MB以上を使う。ここでは重なりなしの
    フレームに対してeinsumで二乗和を求めるので、
    大きな一時配列を作らずに済む。
    """
    hop = HOP_LENGTH
    frame_count = len(y) // hop

    if frame_count < 2:
        return y

    frames = y[: frame_count * hop].reshape(frame_count, hop)

    # einsumは二乗した配列を作らずに行ごとの二乗和を出す。
    energy = np.sqrt(
        np.einsum("ij,ij->i", frames, frames, dtype=np.float64) / hop
    )
    peak = float(energy.max())

    if peak <= 0.0:
        return y

    threshold = peak * (10.0 ** (-float(top_db) / 20.0))
    loud = np.flatnonzero(energy > threshold)

    if loud.size == 0:
        return y

    start = int(loud[0]) * hop
    end = min(len(y), (int(loud[-1]) + 1) * hop)

    return y[start:end]


def estimate_tuning_semitones(harmonic: np.ndarray, sr: int) -> float:
    """
    チューニングのずれを半音単位で推定する。

    36分割のまま推定すると補正範囲が±1/3半音（約±16.7セント）に
    制限され、A=432Hz録音やテープ速度のずれた音源で音名が隣へ滑る。
    """
    try:
        value = float(
            librosa.estimate_tuning(
                y=harmonic,
                sr=sr,
                bins_per_octave=12,
                resolution=0.005,
            )
        )
        return float(np.clip(value, -0.5, 0.5))
    except Exception:
        return 0.0


def place_block_feature(
    destination: np.ndarray,
    block_feature: np.ndarray,
    low: int,
    start: int,
    hop: int,
    block_samples: int,
    reaches_end: bool,
) -> bool:
    """
    のりしろ付きで計算したブロックから、中央の採用部分だけを全体配列へ書く。

    ブロック内フレームkは絶対サンプル low + k * hop に対応する。
    末尾ブロック以外は block_samples 分だけ書き、前後ののりしろは捨てる。
    1次元（音量・オンセット）と2次元（クロマ）の両方を扱う。
    """
    offset_frame = low // hop
    take_start = start // hop - offset_frame
    available = block_feature.shape[-1] - take_start

    if available <= 0:
        return False

    write_start = start // hop
    total = destination.shape[-1]

    if reaches_end:
        length = min(available, total - write_start)
    else:
        length = min(available, total - write_start, block_samples // hop)

    if length <= 0:
        return False

    source = block_feature[..., take_start:take_start + length]
    if destination.ndim == 1:
        destination[write_start:write_start + length] = source
    else:
        destination[:, write_start:write_start + length] = source

    return True


def _parabolic_peak_lag(values: np.ndarray, index: int) -> float:
    """
    ピークとその両隣の3点に放物線を当て、小数精度の頂点位置を返す。

    離散的な自己相関から、フレーム間に落ちる本当のピーク位置を推定する。
    頂点のずれは±0.5ラグに収める（両隣より外れることはないため）。
    """
    if index <= 0 or index >= len(values) - 1:
        return float(index)

    left = float(values[index - 1])
    center = float(values[index])
    right = float(values[index + 1])

    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return float(index)

    offset = 0.5 * (left - right) / denominator
    offset = float(np.clip(offset, -0.5, 0.5))

    return float(index) + offset


def estimate_tempo(
    onset_envelope: np.ndarray,
    sr: int,
    hop: int = ONSET_HOP_LENGTH,
) -> dict[str, float]:
    """
    オンセット包絡の自己相関からテンポ（BPM）を推定する。

    テンポ検出には、実テンポの2倍・半分を答えてしまう「オクターブ誤差」が
    つきまとう。パルス列の自己相関は拍の位置だけでなくその整数倍のラグにも
    山を作るため、自己相関の最大値をそのまま採ると遅い側へ寄りやすい。
    そこで知覚的に自然なテンポ帯を中心にした重みを掛けて代表値を選び、
    もう一方のオクターブ（倍/半）を必ず候補として併記する。

    構成音の判別と違い、倍か半かは音の並びだけでは原理的に確定できない
    （174 BPMのドラムンベースと87 BPMのヒップホップは同じ拍位置になりうる）。
    そのため確定させず、候補と信頼度を添えて利用者に委ねる。
    """
    onset_envelope = np.asarray(onset_envelope, dtype=np.float64)
    empty = {"bpm": 0.0, "alt_bpm": 0.0, "confidence": 0.0}

    if onset_envelope.size < 8:
        return empty

    centered = onset_envelope - onset_envelope.mean()
    if centered.std() < 1e-9:
        return empty

    autocorrelation = np.maximum(
        librosa.autocorrelate(centered, max_size=len(centered)),
        0.0,
    )
    lags = np.arange(1, len(autocorrelation))
    bpms = 60.0 * sr / (hop * lags)

    in_range = (bpms >= TEMPO_MIN_BPM) & (bpms <= TEMPO_MAX_BPM)
    valid_lags = lags[in_range]
    bpms = bpms[in_range]
    strength = autocorrelation[1:][in_range]

    if bpms.size == 0 or strength.max() <= 0.0:
        return empty

    strength = strength / strength.max()

    # ラグは大きいほどBPMが小さいので、補間しやすいよう昇順へ並べ替える。
    order = np.argsort(bpms)
    bpms = bpms[order]
    strength = strength[order]

    def strength_at(bpm: float) -> float:
        if bpm < bpms[0] or bpm > bpms[-1]:
            return 0.0
        return float(np.interp(bpm, bpms, strength))

    def prior(bpm: float) -> float:
        return float(
            np.exp(
                -0.5
                * (np.log2(bpm / TEMPO_PRIOR_CENTER) / TEMPO_PRIOR_SIGMA) ** 2
            )
        )

    # 自己相関は整数ラグ（フレーム単位）でしか値が無いため、BPMが飛び飛びに
    # 量子化される。たとえば130 BPMはラグ20（129.2 BPM）へ丸められ、
    # ハーフの65を2倍しても129のままになる。ピーク前後3点に放物線を当てて
    # 小数ラグの頂点を求め、元の分解能より細かくBPMを復元する。
    peak_lag = int(valid_lags[np.argmax(autocorrelation[valid_lags])])
    refined_lag = _parabolic_peak_lag(autocorrelation, peak_lag)
    raw_peak = float(60.0 * sr / (hop * refined_lag))
    ratios = (1 / 3, 1 / 2, 2 / 3, 1.0, 3 / 2, 2.0, 3.0)
    candidates = sorted(
        {
            round(raw_peak * ratio, 1)
            for ratio in ratios
            if TEMPO_MIN_BPM <= raw_peak * ratio <= TEMPO_MAX_BPM
        }
    )

    scored = sorted(
        ((bpm, strength_at(bpm) * prior(bpm)) for bpm in candidates),
        key=lambda item: item[1],
        reverse=True,
    )

    primary = scored[0][0]

    # 対抗馬は知覚中心の反対側のオクターブ。
    # 遅めに出た代表値（例：88）は速い拍を半分に取った可能性が高いので倍を、
    # 速めに出た代表値には半分を添える。
    if primary <= TEMPO_PRIOR_CENTER:
        alt = round(primary * 2.0, 1)
    else:
        alt = round(primary / 2.0, 1)
    if not (TEMPO_MIN_BPM <= alt <= TEMPO_MAX_BPM):
        alt = 0.0

    confidence = 0.0
    if len(scored) > 1 and scored[0][1] > 0.0:
        confidence = 1.0 - scored[1][1] / scored[0][1]

    return {
        "bpm": float(primary),
        "alt_bpm": float(alt),
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
    }


def fold_to_reference(bpm: float, reference: float) -> float:
    """bpmを2倍/半分してreferenceに最も近いオクターブへ揃える。"""
    if bpm <= 0.0 or reference <= 0.0:
        return bpm

    folded = bpm
    while folded / reference >= 1.4:
        folded /= 2.0
    while folded / reference <= 0.72:
        folded *= 2.0
    return folded


def tempo_curve(
    onset_envelope: np.ndarray,
    sr: int,
    hop: int = ONSET_HOP_LENGTH,
) -> list[dict[str, float]]:
    """窓をずらしながら局所テンポを測り、時間対BPMの並びを返す。"""
    onset_envelope = np.asarray(onset_envelope, dtype=np.float64)
    frames_per_second = sr / hop
    window_frames = int(round(TEMPO_WINDOW_SECONDS * frames_per_second))
    step_frames = max(1, int(round(TEMPO_STEP_SECONDS * frames_per_second)))

    if onset_envelope.size < window_frames:
        return []

    curve: list[dict[str, float]] = []
    last_start = onset_envelope.size - window_frames

    for start in range(0, last_start + 1, step_frames):
        segment = onset_envelope[start:start + window_frames]
        estimate = estimate_tempo(segment, sr, hop)

        if estimate["bpm"] <= 0.0:
            continue

        center_frame = start + window_frames / 2.0
        curve.append(
            {
                "time": center_frame / frames_per_second,
                "bpm": estimate["bpm"],
                "confidence": estimate["confidence"],
            }
        )

    return curve


def _median_smooth(values: list[float], radius: int = 1) -> list[float]:
    smoothed: list[float] = []
    for index in range(len(values)):
        low = max(0, index - radius)
        high = min(len(values), index + radius + 1)
        smoothed.append(float(np.median(values[low:high])))
    return smoothed


def segment_tempo_changes(
    onset_envelope: np.ndarray,
    sr: int,
    duration: float,
    hop: int = ONSET_HOP_LENGTH,
) -> list[dict[str, Any]]:
    """
    局所テンポの並びから、はっきり変わって十分続く境界だけを拾い、
    テンポ区間の一覧を返す。純粋な倍/半の揺れは変化として扱わない。
    """
    curve = tempo_curve(onset_envelope, sr, hop)

    if len(curve) < 3:
        return []

    times = [point["time"] for point in curve]
    smoothed = _median_smooth([point["bpm"] for point in curve], radius=1)

    # 何窓ぶん連続してズレたら「変化」とみなすか。
    sustain_windows = max(
        2,
        int(round(MIN_TEMPO_SEGMENT_SECONDS / TEMPO_STEP_SECONDS)),
    )

    # 貪欲に区間を伸ばし、十分続く逸脱が来たら切る。
    boundaries: list[int] = []  # curveのインデックス（新区間の開始）
    segment_start = 0
    reference = smoothed[0]
    deviating_since: int | None = None

    for index in range(1, len(smoothed)):
        folded = fold_to_reference(smoothed[index], reference)
        relative_gap = abs(folded - reference) / max(reference, 1e-9)

        if relative_gap > TEMPO_CHANGE_RATIO:
            if deviating_since is None:
                deviating_since = index
            sustained = index - deviating_since + 1

            # 逸脱がひとまとまりに続き、なおかつ現区間・新区間とも
            # 最小長を満たせるときだけ境界として確定する。
            new_start = deviating_since
            enough_before = (
                times[new_start] - times[segment_start]
                >= MIN_TEMPO_SEGMENT_SECONDS
            )
            enough_after = (
                duration - times[new_start]
                >= MIN_TEMPO_SEGMENT_SECONDS
            )

            if sustained >= sustain_windows and enough_before and enough_after:
                boundaries.append(new_start)
                segment_start = new_start
                reference = float(np.median(smoothed[new_start:index + 1]))
                deviating_since = None
        else:
            deviating_since = None
            # 現区間の基準を緩やかに更新（新しい値へゆっくり寄せる）
            reference = 0.7 * reference + 0.3 * folded

    # 境界で区間の時間範囲を作り、各区間で改めてテンポを推定する。
    cut_times = [0.0]
    cut_times.extend(times[boundary] for boundary in boundaries)
    cut_times.append(duration)

    frames_per_second = sr / hop
    segments: list[dict[str, Any]] = []

    for index in range(len(cut_times) - 1):
        start_seconds = cut_times[index]
        end_seconds = cut_times[index + 1]
        start_frame = int(round(start_seconds * frames_per_second))
        end_frame = int(round(end_seconds * frames_per_second))
        segment_envelope = onset_envelope[start_frame:end_frame]
        estimate = estimate_tempo(segment_envelope, sr, hop)
        segments.append(
            {
                "start": float(start_seconds),
                "end": float(end_seconds),
                "duration": float(end_seconds - start_seconds),
                **estimate,
            }
        )

    return segments


def block_features(
    harmonic: np.ndarray,
    sr: int,
    tuning: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1ブロック分の全音域クロマ・低音域クロマ・音量を計算する。"""
    cqt = np.abs(
        librosa.cqt(
            harmonic,
            sr=sr,
            hop_length=HOP_LENGTH,
            fmin=librosa.note_to_hz("C1"),
            n_bins=BINS_PER_OCTAVE * N_OCTAVES,
            bins_per_octave=BINS_PER_OCTAVE,
            tuning=tuning,
        )
    )
    compressed_cqt = np.log1p(8.0 * cqt)
    del cqt

    chroma_cqt = librosa.feature.chroma_cqt(
        C=compressed_cqt,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_chroma=12,
        n_octaves=N_OCTAVES,
        bins_per_octave=BINS_PER_OCTAVE,
        tuning=tuning,
    )
    chroma_cens = librosa.feature.chroma_cens(
        C=compressed_cqt,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_chroma=12,
        n_octaves=N_OCTAVES,
        bins_per_octave=BINS_PER_OCTAVE,
        tuning=tuning,
        win_len_smooth=21,
    )

    frame_count = min(chroma_cqt.shape[1], chroma_cens.shape[1])
    full_chroma = (
        0.72 * chroma_cqt[:, :frame_count]
        + 0.28 * chroma_cens[:, :frame_count]
    )

    # C1〜B3の3オクターブを低音特徴として集計。
    # ビン0がC1のちょうど中心なので、素直に3本ずつ束ねると
    # 各音名が「中心＋上寄り2本」になり、上隣の音名の裾を吸い込む。
    # 半音の中心を挟むように1本ずらしてから束ねる。
    bins_per_semitone = BINS_PER_OCTAVE // 12
    bass_cqt = compressed_cqt[: 3 * BINS_PER_OCTAVE, :frame_count]
    bass_cqt = np.roll(bass_cqt, bins_per_semitone // 2, axis=0)
    bass_chroma = bass_cqt.reshape(
        3,
        12,
        bins_per_semitone,
        frame_count,
    ).sum(axis=(0, 2))

    block_rms = librosa.feature.rms(
        y=harmonic,
        frame_length=4096,
        hop_length=HOP_LENGTH,
    )[0, :frame_count]

    return full_chroma, bass_chroma, block_rms


def extract_features(
    y: np.ndarray,
    sr: int,
    progress_callback: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    音声をブロックへ区切りながら特徴量を組み立てる。

    HPSSとCQTはブロック内で完結させ、外へ持ち出すのはクロマ・音量・
    オンセット包絡だけ。こうすると必要なメモリが曲の長さでほぼ変わらない。
    ブロック境界の歪みを避けるため前後にのりしろを付けて計算し、
    採用するのは中央部分だけにする。

    クロマ・音量はHOP_LENGTH刻み、テンポ用のオンセット包絡は
    ONSET_HOP_LENGTH刻みで、それぞれ別の全体配列へ貼り合わせる。
    """
    n_samples = len(y)
    total_frames = 1 + n_samples // HOP_LENGTH
    total_onset_frames = 1 + n_samples // ONSET_HOP_LENGTH

    block_samples = int(round(BLOCK_SECONDS * sr / HOP_LENGTH)) * HOP_LENGTH
    pad_samples = int(round(BLOCK_PAD_SECONDS * sr / HOP_LENGTH)) * HOP_LENGTH

    full_chroma = np.zeros((12, total_frames), dtype=np.float64)
    bass_chroma = np.zeros((12, total_frames), dtype=np.float64)
    rms = np.zeros(total_frames, dtype=np.float64)
    onset_envelope = np.zeros(total_onset_frames, dtype=np.float64)

    tuning_semitones: float | None = None
    filled_any = False
    block_count = max(1, math.ceil(n_samples / block_samples))

    for block_index, start in enumerate(
        range(0, n_samples, block_samples)
    ):
        safe_progress(
            progress_callback,
            0.14 + 0.22 * (block_index / block_count),
            f"特徴量を計算中…（{block_index + 1}/{block_count}）",
        )

        low = max(0, start - pad_samples)
        high = min(n_samples, start + block_samples + pad_samples)
        segment = y[low:high]

        if len(segment) < HOP_LENGTH * 4:
            continue

        reaches_end = high >= n_samples

        # オンセット包絡は打楽器を含む生の信号から取る（HPSS前）。
        block_onset = librosa.onset.onset_strength(
            y=segment,
            sr=sr,
            hop_length=ONSET_HOP_LENGTH,
        )

        harmonic = librosa.effects.harmonic(segment, margin=3.0)

        if tuning_semitones is None:
            tuning_semitones = estimate_tuning_semitones(harmonic, sr)

        # librosa.cqtのtuningは「そのCQTのビン幅」を単位に取るため、
        # 半音単位の推定値をビン単位へ変換して渡す。
        block_full, block_bass, block_rms = block_features(
            harmonic,
            sr,
            tuning_semitones * (BINS_PER_OCTAVE / 12.0),
        )
        del harmonic, segment

        placed = place_block_feature(
            full_chroma, block_full, low, start,
            HOP_LENGTH, block_samples, reaches_end,
        )
        place_block_feature(
            bass_chroma, block_bass, low, start,
            HOP_LENGTH, block_samples, reaches_end,
        )
        place_block_feature(
            rms, block_rms, low, start,
            HOP_LENGTH, block_samples, reaches_end,
        )
        place_block_feature(
            onset_envelope, block_onset, low, start,
            ONSET_HOP_LENGTH, block_samples, reaches_end,
        )
        filled_any = filled_any or placed

    if not filled_any:
        raise ValueError("和音・旋律成分を十分に検出できませんでした。")

    full_chroma /= np.sum(full_chroma, axis=0, keepdims=True) + 1e-12
    bass_chroma /= np.sum(bass_chroma, axis=0, keepdims=True) + 1e-12

    return (
        full_chroma,
        bass_chroma,
        rms,
        onset_envelope,
        float(tuning_semitones or 0.0),
    )


def analyze_audio_file(
    audio_path: str | Path,
    sensitivity: str = "標準",
    notation: str = "♯優先",
    progress_callback: Callable | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not audio_path:
        raise ValueError("音声ファイルを選択してください。")

    if sensitivity not in SENSITIVITY_SETTINGS:
        sensitivity = "標準"
    if notation not in ("♯優先", "♭優先"):
        notation = "♯優先"

    settings = SENSITIVITY_SETTINGS[sensitivity]
    audio_path = str(audio_path)
    suffix = Path(audio_path).suffix.lower().lstrip(".") or "?"

    safe_progress(progress_callback, 0.02, "音声情報を確認中…")
    try:
        original_duration = float(librosa.get_duration(path=audio_path))
    except Exception as error:  # noqa: BLE001
        raise _decode_error(suffix) from error

    if original_duration < MIN_AUDIO_SECONDS:
        raise ValueError(
            f"{MIN_AUDIO_SECONDS:.0f}秒以上の音声を使ってください。"
        )
    if original_duration > MAX_DURATION_SECONDS:
        raise ValueError("現在の試作版では20分以内の音声に対応しています。")

    safe_progress(progress_callback, 0.07, "音声を読み込み中…")
    try:
        y, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True,
        )
    except Exception as error:  # noqa: BLE001
        raise _decode_error(suffix) from error

    if y.size == 0 or not np.any(np.isfinite(y)):
        raise ValueError("音声を正常に読み込めませんでした。")

    y = np.nan_to_num(y, copy=False)

    y = trim_silence(y, top_db=42.0)
    duration = len(y) / sr

    if duration < MIN_AUDIO_SECONDS:
        raise ValueError("無音部分を除くと音声が短すぎます。")

    safe_progress(progress_callback, 0.14, "打楽器成分を抑制中…")
    (
        full_chroma,
        bass_chroma,
        rms,
        onset_envelope,
        tuning_semitones,
    ) = extract_features(y, sr, progress_callback)

    # 音声そのものはもう使わないので手放す。
    del y
    gc.collect()

    if not np.any(rms > 1e-8):
        raise ValueError("和音・旋律成分を十分に検出できませんでした。")

    tempo_segments = segment_tempo_changes(onset_envelope, sr, duration)
    if not tempo_segments:
        tempo_segments = [
            {
                "start": 0.0,
                "end": duration,
                "duration": duration,
                **estimate_tempo(onset_envelope, sr),
            }
        ]

    # 見出しのBPMは最も長く続いたテンポ区間を採用する。
    headline_tempo = max(tempo_segments, key=lambda seg: seg["duration"])
    tempo = {
        "bpm": headline_tempo["bpm"],
        "alt_bpm": headline_tempo["alt_bpm"],
        "confidence": headline_tempo["confidence"],
    }
    tempo_changed = len(tempo_segments) > 1

    safe_progress(progress_callback, 0.38, "半音移動をスキャン中…")
    raw_candidates = scan_change_candidates(
        full_chroma,
        bass_chroma,
        rms,
        sr,
        duration,
        settings,
    )
    selected_candidates = select_candidate_peaks(
        raw_candidates,
        settings,
    )

    safe_progress(progress_callback, 0.55, "転調位置を細かく調整中…")
    refined_boundaries = [
        refine_boundary(
            candidate,
            full_chroma,
            bass_chroma,
            rms,
            sr,
            duration,
        )
        for candidate in selected_candidates
    ]
    refined_boundaries = deduplicate_refined_boundaries(
        refined_boundaries,
        float(settings["minimum_boundary_gap"]),
    )

    # 候補はやや広めに残し、全区間の状態追跡で採否を決める。
    # ただし信頼度だけの抜け道は作らない。以前はここで
    # boundary_confidence >= 38 を通していたため、shift_aware_boundary_pass
    # の5度方向チェックが実質無効になり、Ⅳ・Ⅴを強調する進行が
    # そのまま転調候補として通過していた。
    minimum_scale_change = float(settings["minimum_scale_change"])

    def confidence_fallback(boundary: dict[str, Any]) -> bool:
        if boundary_confidence(boundary) < 52.0:
            return False

        scale_evidence = float(
            boundary.get("scale_change_evidence", 0.0)
        )

        if int(boundary.get("shift", 0)) % 12 in FIFTH_SHIFTS:
            return scale_evidence >= minimum_scale_change * 1.3

        return scale_evidence >= minimum_scale_change * 0.5

    candidate_boundaries = [
        boundary
        for boundary in refined_boundaries
        if (
            shift_aware_boundary_pass(boundary, settings)
            or confidence_fallback(boundary)
        )
    ]

    segment_ranges = make_segment_ranges(duration, candidate_boundaries)

    safe_progress(progress_callback, 0.68, "複数転調の状態列を解析中…")
    segment_results = [
        segment_key_analysis(
            full_chroma,
            bass_chroma,
            rms,
            sr,
            start_seconds,
            end_seconds,
        )
        for start_seconds, end_seconds in segment_ranges
    ]

    state_path = optimize_segment_key_sequence(
        segment_results,
        candidate_boundaries,
        settings,
    )
    state_path = stabilize_short_state_excursions(
        state_path,
        segment_results,
        candidate_boundaries,
        settings,
    )
    state_path = suppress_fifth_round_trips(
        state_path,
        segment_results,
        candidate_boundaries,
    )

    final_segments, boundaries, suppressed_boundaries = (
        merge_optimized_state_runs(
            segment_results,
            candidate_boundaries,
            state_path,
            full_chroma,
            bass_chroma,
            rms,
            sr,
        )
    )

    main_key_index, main_confidence, main_scores, main_duration_ratio = (
        aggregate_main_key(final_segments)
    )

    start_key_index = int(final_segments[0]["final_key"])
    modulation_label, modulation_confidence = overall_modulation_confidence(
        boundaries
    )

    main_ranking = np.argsort(main_scores)[::-1]
    second_key_index = int(main_ranking[1])

    boundary_lines: list[str] = []
    seen_keys = {start_key_index}
    return_count = 0

    for index, boundary in enumerate(boundaries):
        before_key = int(final_segments[index]["final_key"])
        after_key = int(final_segments[index + 1]["final_key"])

        if after_key == start_key_index and before_key != start_key_index:
            transition_type = "元キーへ復帰"
            return_count += 1
        elif after_key in seen_keys:
            transition_type = "過去のキーへ復帰"
            return_count += 1
        else:
            transition_type = "新しい転調"

        boundary["transition_type"] = transition_type
        seen_keys.add(after_key)

        boundary_lines.append(
            f'・{format_seconds(boundary["time"])}頃: '
            f'{key_name(before_key, notation)} → '
            f'{key_name(after_key, notation)} '
            f'（{format_shift(boundary["shift"])}／{transition_type}）'
        )

    transition_summary = (
        "<br>".join(boundary_lines)
        if boundary_lines
        else "明確な半音移動を伴う転調は検出されませんでした。"
    )

    if tempo_changed:
        tempo_row_lines = []
        for index, seg in enumerate(tempo_segments):
            start_text = format_seconds(seg["start"])
            end_text = format_seconds(seg["end"])
            alt_text = f'{seg["alt_bpm"]:.0f} BPM' if seg["alt_bpm"] else "—"
            confidence_text = f'{seg["confidence"] * 100:.0f}%'
            tempo_row_lines.append(
                f'| {index + 1} | {start_text}〜{end_text} '
                f'| {seg["bpm"]:.0f} BPM | {alt_text} | {confidence_text} |'
            )
        tempo_rows = "\n".join(tempo_row_lines)
        tempo_section = (
            "\n\n### テンポの変化\n\n"
            f"曲中で **{len(tempo_segments)}個** のテンポ区間を検出しました。\n\n"
            "| 区間 | 時間 | BPM | 倍/半の候補 | 信頼度 |\n"
            "|---|---|---|---|---|\n"
            f"{tempo_rows}\n\n"
            "> ハーフタイム／ダブルタイム（拍の取り方が2倍・半分になるだけ）は、"
            "同じテンポとして扱い、変化には数えていません。"
        )
    else:
        tempo_section = ""

    relative_index = relative_key_index(main_key_index)
    relative_gap = float(
        main_scores[main_key_index] - main_scores[relative_index]
    )

    if relative_gap < 0.035:
        relative_note = (
            f"{key_name(relative_index, notation)}とのスコア差が小さく、"
            "相対長調・短調の判別は曖昧です。"
        )
    else:
        relative_note = "相対長調・短調との区別は比較的安定しています。"

    main_key_text = key_name(main_key_index, notation)
    start_key_text = key_name(start_key_index, notation)

    if boundaries:
        last_key_index = int(final_segments[-1]["final_key"])
        final_key_row = (
            f"| 最終区間のキー | "
            f"{key_name(last_key_index, notation)} / "
            f"{camelot_code(last_key_index)} |\n"
        )
    else:
        final_key_row = ""

    result_markdown = f"""
## 判定結果

| 項目 | 結果 |
|---|---|
| **推定主調** | **{main_key_text}** |
| **Camelot** | **{camelot_code(main_key_index)}** |
| **推定BPM** | **{format_bpm(tempo)}**{"（テンポ変化あり）" if tempo_changed else ""} |
| BPMの信頼度 | {tempo["confidence"] * 100:.0f}%（{bpm_confidence_label(tempo["confidence"])}） |
| **参考信頼度** | **{main_confidence:.0f}%（{confidence_label(main_confidence)}）** |
| 第2候補 | {key_name(second_key_index, notation)} / {camelot_code(second_key_index)} |
| 開始時のキー | {start_key_text} / {camelot_code(start_key_index)} |
{final_key_row}| **転調の可能性** | **{modulation_label}（{modulation_confidence:.0f}%）** |
| 主調の総滞在率 | {main_duration_ratio * 100:.0f}% |
| 解析時間 | {format_seconds(duration)} |
| 検出区間数 | {len(final_segments)} |
| 検出した転調回数 | {len(boundaries)}回 |
| キーへの復帰 | {return_count}回 |
| 抑制した疑似転調 | {len(suppressed_boundaries)}件 |
| 転調感度 | {sensitivity} |

### 検出した転調

{transition_summary}
{tempo_section}

### 判定方法について

転調候補を並べたあと、曲全体を24調の**状態遷移**として最適化しています。
各区間のキーらしさと境界の移調量を同時に評価するため、
`元キー → 落ちサビのキー → 元キー`の復帰や、複数回の転調を追跡できます。

転調の判定には**調号そのものが動いたか**を必ず確認しています。
たとえばC MajorからG Majorへ本当に転調していればF♯が現れてFが減りますが、
サビでⅣ・Ⅴを強調しているだけの進行では構成音は変わりません。
この差を見ることで、コード進行の重心移動を転調と誤認しにくくしています。

### 相対調の判別

{relative_note}

> 参考信頼度は正解確率ではありません。候補間の差や調性感をまとめた目安です。
> 一時的な借用和音やコード変化は、転調として扱わないよう保守的に判定しています。
""".strip()

    segment_table = pd.DataFrame(
        [
            {
                "区間": index + 1,
                "開始": format_seconds(segment["start"]),
                "終了": format_seconds(segment["end"]),
                "長さ": format_seconds(segment["duration"]),
                "推定キー": key_name(segment["final_key"], notation),
                "Camelot": camelot_code(segment["final_key"]),
                "直接判定": key_name(segment["absolute_key"], notation),
                "直接判定の信頼度": round(
                    float(segment["absolute_confidence"]),
                    1,
                ),
                "採用方法": segment["method"],
            }
            for index, segment in enumerate(final_segments)
        ]
    )

    boundary_table = pd.DataFrame(
        [
            {
                "転調位置": format_seconds(boundary["time"]),
                "推定移調量": format_shift(boundary["shift"]),
                "転調前": key_name(
                    final_segments[index]["final_key"],
                    notation,
                ),
                "転調後": key_name(
                    final_segments[index + 1]["final_key"],
                    notation,
                ),
                "位置信頼度": round(boundary_confidence(boundary), 1),
                "移調一致度": round(
                    100.0 * boundary.get(
                        "best_nonzero_alignment",
                        boundary.get("best_nonzero_similarity", 0.0),
                    ),
                    1,
                ),
                "0半音との差": round(
                    100.0 * boundary.get(
                        "hybrid_gain",
                        boundary.get("gain", 0.0),
                    ),
                    1,
                ),
                "調号の変化": round(
                    100.0 * boundary.get("scale_change_evidence", 0.0),
                    1,
                ),
                "種類": boundary.get(
                    "transition_type",
                    "新しい転調",
                ),
                "検出経路": boundary.get(
                    "detection_route",
                    "ハイブリッド",
                ),
            }
            for index, boundary in enumerate(boundaries)
        ],
        columns=[
            "転調位置",
            "推定移調量",
            "転調前",
            "転調後",
            "位置信頼度",
            "移調一致度",
            "0半音との差",
            "調号の変化",
            "種類",
            "検出経路",
        ],
    )

    candidate_table = pd.DataFrame(
        [
            {
                "順位": rank,
                "キー": key_name(int(candidate_index), notation),
                "Camelot": camelot_code(int(candidate_index)),
                "スコア": round(
                    float(
                        main_scores[int(candidate_index)]
                    ),
                    4,
                ),
                "主調との差": round(
                    float(
                        main_scores[main_key_index]
                        - main_scores[int(candidate_index)]
                    ),
                    4,
                ),
            }
            for rank, candidate_index in enumerate(
                main_ranking[:8],
                start=1,
            )
        ]
    )

    details = {
        "ファイル名": Path(audio_path).name,
        "推定主調": main_key_text,
        "開始キー": start_key_text,
        "推定BPM": round(tempo["bpm"], 1),
        "BPMの倍半候補": round(tempo["alt_bpm"], 1),
        "BPMの信頼度": round(tempo["confidence"] * 100.0, 1),
        "テンポ変化": tempo_changed,
        "テンポ区間数": len(tempo_segments),
        "テンポ区間": [
            {
                "開始": format_seconds(seg["start"]),
                "終了": format_seconds(seg["end"]),
                "BPM": round(seg["bpm"], 1),
                "倍半候補": round(seg["alt_bpm"], 1),
                "信頼度": round(seg["confidence"] * 100.0, 1),
            }
            for seg in tempo_segments
        ],
        "参考信頼度": round(main_confidence, 1),
        "転調の可能性": modulation_label,
        "転調信頼度": round(modulation_confidence, 1),
        "検出転調数": len(boundaries),
        "チューニング補正値（セント）": round(tuning_semitones * 100.0, 1),
        "転調候補走査数": len(raw_candidates),
        "採用した境界数": len(boundaries),
        "キーへの復帰回数": return_count,
        "主調の総滞在率": round(main_duration_ratio * 100, 1),
        "状態追跡候補数": len(candidate_boundaries),
        "抑制した疑似転調数": len(suppressed_boundaries),
        "抑制した疑似転調": [
            {
                "位置": format_seconds(boundary["time"]),
                "移調量": format_shift(boundary.get("shift", 0)),
                "理由": boundary.get(
                    "suppression_reason",
                    "状態追跡で前後が同一キーになった",
                ),
            }
            for boundary in suppressed_boundaries
        ],
    }

    safe_progress(progress_callback, 1.0, "解析完了！")

    return (
        result_markdown,
        segment_table,
        boundary_table,
        candidate_table,
        details,
    )


print("Aryth Key Finder v0.5.2 Beta engine ready.")
