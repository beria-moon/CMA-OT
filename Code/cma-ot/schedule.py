from __future__ import annotations

import math


def cosine_anneal(epoch: int, start_epoch: int, end_epoch: int, start_value: float, end_value: float) -> float:
    """Cosine interpolation that equals ``start_value`` and ``end_value`` at the boundaries."""
    if end_epoch <= start_epoch:
        raise ValueError("end_epoch must be greater than start_epoch")
    if epoch <= start_epoch:
        return start_value
    if epoch >= end_epoch:
        return end_value
    progress = (epoch - start_epoch) / (end_epoch - start_epoch)
    return start_value + 0.5 * (end_value - start_value) * (1.0 - math.cos(math.pi * progress))


def curriculum_weights(epoch: int, phase1_end: int = 30, phase2_end: int = 100, transition_len: int = 30) -> dict[str, float]:
    """Paper three-stage cosine curriculum for CMA-OT layer supervision.

    Top:    1.0 -> 0.4 over epochs 30-60, then 0.4 -> 0.2 over 100-130.
    Middle: 0.0 -> 0.6 over epochs 30-60, then 0.6 -> 0.3 over 100-130.
    Bottom: 0.0 -> 0.5 over epochs 100-130.

    Returned weights are normalized to sum to one.
    """
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if phase2_end < phase1_end + transition_len:
        raise ValueError("phase2_end must not overlap the first transition")
    if transition_len <= 0:
        raise ValueError("transition_len must be positive")

    first_transition_end = phase1_end + transition_len
    second_transition_end = phase2_end + transition_len

    if epoch < phase1_end:
        top, middle = 1.0, 0.0
    elif epoch < first_transition_end:
        top = cosine_anneal(epoch, phase1_end, first_transition_end, 1.0, 0.4)
        middle = cosine_anneal(epoch, phase1_end, first_transition_end, 0.0, 0.6)
    elif epoch < phase2_end:
        top, middle = 0.4, 0.6
    elif epoch < second_transition_end:
        top = cosine_anneal(epoch, phase2_end, second_transition_end, 0.4, 0.2)
        middle = cosine_anneal(epoch, phase2_end, second_transition_end, 0.6, 0.3)
    else:
        top, middle = 0.2, 0.3

    bottom = 0.0 if epoch < phase2_end else cosine_anneal(epoch, phase2_end, second_transition_end, 0.0, 0.5)
    total = top + middle + bottom
    return {"top": top / total, "middle": middle / total, "bottom": bottom / total}
