"""Aryth Key Finder v0.4 Beta — analysis engine."""

from __future__ import annotations

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
BINS_PER_OCTAVE = 36
N_OCTAVES = 7
MAX_DURATION_SECONDS = 20 * 60
MIN_AUDIO_SECONDS = 12.0

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
        "transition_penalty": 1.02,
    },
    "標準": {
        "candidate_threshold": 0.46,
        "minimum_gain": 0.010,
        "minimum_structure": 0.25,
        "minimum_segment_seconds": 14.0,
        "minimum_boundary_gap": 14.0,
        "transition_penalty": 0.84,
    },
    "高め（短い転調も拾う）": {
        "candidate_threshold": 0.36,
        "minimum_gain": 0.004,
        "minimum_structure": 0.18,
        "minimum_segment_seconds": 10.0,
        "minimum_boundary_gap": 10.0,
        "transition_penalty": 0.68,
    },
}


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
BASS_TEMPLATES = _build_templates(BASS_MAJOR, BASS_MINOR)


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
    従来の全音域判定を主役にし、曖昧な場合だけ低音域の情報を強める。
    転調しない曲での既存精度を崩しにくい設計。
    """
    krumhansl_scores = profile_scores(full_chroma, KRUMHANSL_TEMPLATES)
    temperley_scores = profile_scores(full_chroma, TEMPERLEY_TEMPLATES)
    base_scores = 0.76 * krumhansl_scores + 0.24 * temperley_scores

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

    combined_alignment = (
        0.56 * chroma_alignment
        + 0.44 * key_alignment
    )

    nonzero_ranking = np.argsort(combined_alignment[1:])[::-1] + 1
    best_nonzero_shift = int(nonzero_ranking[0])

    same_alignment = float(combined_alignment[0])
    best_nonzero_alignment = float(
        combined_alignment[best_nonzero_shift]
    )
    hybrid_gain = best_nonzero_alignment - same_alignment

    before_winner = int(np.argmax(before_scores))
    after_winner = int(np.argmax(after_scores))
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

    shift_strength = float(
        np.clip(
            0.48 * np.clip(
                (hybrid_gain - 0.002) / 0.105,
                0.0,
                1.0,
            )
            + 0.23 * np.clip(
                (best_nonzero_alignment - 0.54) / 0.38,
                0.0,
                1.0,
            )
            + 0.29 * structural_strength,
            0.0,
            1.0,
        )
    )

    return {
        **chroma_result,
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

    same_alignment = float(combined_alignment[0])
    best_nonzero_alignment = float(
        combined_alignment[chosen_shift]
    )

    return {
        **strongest,
        "shift": chosen_shift,
        "best_nonzero_shift": chosen_shift,
        "combined_alignment": combined_alignment,
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

        if primary_pass or structural_fallback:
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
        return bool(
            quality >= 76.0
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
            alignment >= 0.52
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

    return bool(base_primary or base_structural)


def suppress_fifth_round_trips(
    boundaries: list[dict[str, Any]],
    segment_results: list[dict[str, Any]],
    sensitivity: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    -5半音の後に+5半音など、5度方向へ移動して元へ戻る組を検出する。
    根拠が非常に強い場合を除き、機能和声・コード進行による疑似転調として抑制する。
    """
    if len(boundaries) < 2:
        return list(boundaries), []

    remove_indices: set[int] = set()
    suppressed: list[dict[str, Any]] = []

    for index in range(len(boundaries) - 1):
        first = boundaries[index]
        second = boundaries[index + 1]

        first_shift = int(first.get("shift", 0)) % 12
        second_shift = int(second.get("shift", 0)) % 12

        is_inverse_fifth_pair = (
            first_shift in FIFTH_SHIFTS
            and second_shift in FIFTH_SHIFTS
            and (first_shift + second_shift) % 12 == 0
        )

        if not is_inverse_fifth_pair:
            continue

        middle_segment = (
            segment_results[index + 1]
            if index + 1 < len(segment_results)
            else None
        )
        middle_confidence = (
            float(middle_segment["absolute_confidence"])
            if middle_segment is not None
            else 0.0
        )

        first_quality = boundary_confidence(first)
        second_quality = boundary_confidence(second)
        first_consensus = float(first.get("consensus", 0.0))
        second_consensus = float(second.get("consensus", 0.0))
        first_structure = float(
            first.get("structural_strength", 0.0)
        )
        second_structure = float(
            second.get("structural_strength", 0.0)
        )

        both_direct_routes = (
            first.get("detection_route") == "移調量"
            and second.get("detection_route") == "移調量"
        )

        # 本当に「転調して元へ戻った」とみなすには、
        # 両境界と中間区間のすべてにかなり強い証拠を要求する。
        exceptionally_strong = (
            min(first_quality, second_quality) >= 91.0
            and min(first_consensus, second_consensus) >= 0.74
            and min(first_structure, second_structure) >= 0.54
            and middle_confidence >= 72.0
            and both_direct_routes
        )

        if not exceptionally_strong:
            remove_indices.update({index, index + 1})
            suppressed.extend([first, second])

    kept = [
        boundary
        for index, boundary in enumerate(boundaries)
        if index not in remove_indices
    ]

    return kept, suppressed




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


def choose_chained_segment_keys(
    segment_results: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not segment_results:
        return []

    final_results: list[dict[str, Any]] = []
    first = dict(segment_results[0])
    first["final_key"] = int(first["absolute_key"])
    first["method"] = "区間を直接判定"
    first["shift_from_previous"] = None
    first["shift_support"] = None
    final_results.append(first)

    for index in range(1, len(segment_results)):
        current = dict(segment_results[index])
        previous = final_results[index - 1]
        boundary = boundaries[index - 1]

        shift = int(boundary["shift"])
        shifted_same_mode = transpose_key(
            previous["final_key"],
            shift,
        )
        shifted_relative = relative_key_index(shifted_same_mode)

        # 原則として転調前と同じメジャー/マイナーを維持する。
        # 相対調側が明確に強い場合だけモード変更を許可する。
        same_mode_score = float(
            current["scores"][shifted_same_mode]
        )
        relative_mode_score = float(
            current["scores"][shifted_relative]
        )

        if (
            relative_mode_score - same_mode_score >= 0.085
            and current["absolute_confidence"] >= 58.0
        ):
            shifted_candidate = shifted_relative
        else:
            shifted_candidate = shifted_same_mode

        absolute_candidate = int(current["absolute_key"])
        shifted_score = float(current["scores"][shifted_candidate])
        absolute_score = float(current["scores"][absolute_candidate])
        score_gap = absolute_score - shifted_score

        shift_support = float(
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

        if (
            shift_support >= 0.62
            and structural_strength >= 0.28
            and score_gap <= 0.16
        ):
            final_key = shifted_candidate
            method = (
                f"前区間の同系統から{format_shift(shift)}"
            )
        elif (
            hybrid_gain >= 0.006
            and score_gap <= 0.090
        ):
            final_key = shifted_candidate
            method = (
                f"移調量を優先（{format_shift(shift)}）"
            )
        else:
            final_key = absolute_candidate
            method = "区間を直接判定"

        current["final_key"] = int(final_key)
        current["method"] = method
        current["shift_from_previous"] = shift
        current["shift_support"] = shift_support
        current["shift_score_gap"] = float(score_gap)
        final_results.append(current)

    return final_results



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

        transition_matrix = np.full(
            (n_states, n_states),
            -base_penalty,
            dtype=np.float64,
        )

        for previous_key in range(n_states):
            previous_tonic, _ = KEY_LABELS[previous_key]

            for current_key in range(n_states):
                if current_key == previous_key:
                    # 弱い候補では同じキーを維持し、強い候補では維持を少し不利にする。
                    transition_matrix[previous_key, current_key] = (
                        0.22 * (1.0 - quality)
                        - 0.62 * quality * max(structural, 0.15)
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
                expected_bonus = 0.56 if shift == expected_shift else 0.0
                mode_score = _mode_relation_score(
                    previous_key,
                    current_key,
                    shift,
                )

                fifth_penalty = 0.0
                if shift in FIFTH_SHIFTS and quality < 0.84:
                    fifth_penalty = 0.25 + 0.20 * (0.84 - quality)

                transition_matrix[previous_key, current_key] = (
                    -base_penalty
                    + 1.42 * quality * alignment_support
                    + expected_bonus * quality
                    + 0.48 * structural
                    + 0.18 * consensus
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
        left_quality = boundary_confidence(candidate_boundaries[index - 1])
        right_quality = boundary_confidence(candidate_boundaries[index])

        weak_excursion = (
            segment["duration"] < threshold
            and segment["absolute_confidence"] < 48.0
            and min(left_quality, right_quality) < 58.0
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

    runs: list[tuple[int, int, int]] = []
    run_start = 0
    run_state = int(state_path[0])

    for index in range(1, len(state_path)):
        current_state = int(state_path[index])
        if current_state != run_state:
            runs.append((run_start, index - 1, run_state))
            run_start = index
            run_state = current_state

    runs.append((run_start, len(state_path) - 1, run_state))

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

    main_key = max(duration_by_key, key=duration_by_key.get)
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

    safe_progress(progress_callback, 0.02, "音声情報を確認中…")
    original_duration = float(librosa.get_duration(path=audio_path))

    if original_duration < MIN_AUDIO_SECONDS:
        raise ValueError(
            f"{MIN_AUDIO_SECONDS:.0f}秒以上の音声を使ってください。"
        )
    if original_duration > MAX_DURATION_SECONDS:
        raise ValueError("現在の試作版では20分以内の音声に対応しています。")

    safe_progress(progress_callback, 0.07, "音声を読み込み中…")
    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if y.size == 0 or not np.any(np.isfinite(y)):
        raise ValueError("音声を正常に読み込めませんでした。")

    y = np.nan_to_num(y, copy=False)
    y, _ = librosa.effects.trim(y, top_db=42)
    duration = len(y) / sr

    if duration < MIN_AUDIO_SECONDS:
        raise ValueError("無音部分を除くと音声が短すぎます。")

    safe_progress(progress_callback, 0.14, "打楽器成分を抑制中…")
    harmonic = librosa.effects.harmonic(y, margin=3.0)

    if not np.any(np.abs(harmonic) > 1e-8):
        raise ValueError("和音・旋律成分を十分に検出できませんでした。")

    tuning_audio = harmonic[: min(len(harmonic), sr * 180)]
    try:
        tuning = float(
            librosa.estimate_tuning(
                y=tuning_audio,
                sr=sr,
                bins_per_octave=BINS_PER_OCTAVE,
                resolution=0.01,
            )
        )
        tuning = float(np.clip(tuning, -0.5, 0.5))
    except Exception:
        tuning = 0.0

    safe_progress(progress_callback, 0.23, "全音域と低音域の特徴を計算中…")
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
    chroma_cqt = chroma_cqt[:, :frame_count]
    chroma_cens = chroma_cens[:, :frame_count]

    full_chroma = 0.72 * chroma_cqt + 0.28 * chroma_cens
    full_chroma /= np.sum(full_chroma, axis=0, keepdims=True) + 1e-12

    # C1〜B3の3オクターブを低音特徴として集計
    bass_bin_count = 3 * BINS_PER_OCTAVE
    bass_cqt = compressed_cqt[:bass_bin_count, :frame_count]
    bass_chroma = bass_cqt.reshape(
        3,
        12,
        BINS_PER_OCTAVE // 12,
        frame_count,
    ).sum(axis=(0, 2))
    bass_chroma /= np.sum(bass_chroma, axis=0, keepdims=True) + 1e-12

    rms = librosa.feature.rms(
        y=harmonic,
        frame_length=4096,
        hop_length=HOP_LENGTH,
    )[0, :frame_count]

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
    candidate_boundaries = [
        boundary
        for boundary in refined_boundaries
        if (
            shift_aware_boundary_pass(boundary, settings)
            or boundary_confidence(boundary) >= 38.0
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

### 判定方法について

転調候補を並べたあと、曲全体を24調の**状態遷移**として最適化しています。
各区間のキーらしさと境界の移調量を同時に評価するため、
`元キー → 落ちサビのキー → 元キー`の復帰や、複数回の転調を追跡できます。

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
        "参考信頼度": round(main_confidence, 1),
        "転調の可能性": modulation_label,
        "転調信頼度": round(modulation_confidence, 1),
        "検出転調数": len(boundaries),
        "チューニング補正値": round(tuning, 4),
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


print("Aryth Key Finder v0.4 Beta engine ready.")
