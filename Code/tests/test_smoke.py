import math

import torch

from cma_ot.dance_encoder import DanceRhythmEncoder
from cma_ot.schedule import cosine_anneal, curriculum_weights


def test_dance_encoder_shape():
    encoder = DanceRhythmEncoder(num_joints=17, output_dim=512)
    output = encoder(torch.randn(2, 300, 17, 3), target_frames=108)
    assert output.shape == (2, 108, 512)


def test_cosine_anneal_boundaries_and_midpoint():
    assert cosine_anneal(30, 30, 60, 1.0, 0.4) == 1.0
    assert cosine_anneal(60, 30, 60, 1.0, 0.4) == 0.4
    assert math.isclose(cosine_anneal(45, 30, 60, 1.0, 0.4), 0.7)


def test_paper_curriculum_stages():
    assert curriculum_weights(0) == {"top": 1.0, "middle": 0.0, "bottom": 0.0}
    assert curriculum_weights(30) == {"top": 1.0, "middle": 0.0, "bottom": 0.0}
    assert curriculum_weights(60) == {"top": 0.4, "middle": 0.6, "bottom": 0.0}
    assert curriculum_weights(100) == {"top": 0.4, "middle": 0.6, "bottom": 0.0}
    assert curriculum_weights(130) == {"top": 0.2, "middle": 0.3, "bottom": 0.5}
    assert all(math.isclose(sum(curriculum_weights(epoch).values()), 1.0) for epoch in (0, 45, 60, 100, 115, 130, 199))
