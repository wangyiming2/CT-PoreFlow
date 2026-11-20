from __future__ import annotations

import torch

from src.models import (
    GeometryTimeTransformer,
    GeometryDeepONet,
    QueryTimeConvLSTM,
    SteadyResidualHybridUNet,
    TimeConditionedFNO,
    TimeConditionedUNet,
)


def test_output_shape_and_solid_mask() -> None:
    geometry = torch.rand(2, 3, 100, 100)
    geometry[:, 0] = (geometry[:, 0] > 0.4).float()
    time = torch.tensor([0.0, 1.0])
    for model in [
        TimeConditionedUNet(base=8),
        TimeConditionedFNO(width=8, modes=4, layers=2),
        GeometryDeepONet(basis=8, branch_width=4, trunk_width=16),
        QueryTimeConvLSTM(base=4, hidden_channels=8, recurrent_steps=2),
        GeometryTimeTransformer(d_model=16, heads=4, layers=1, patch_size=10),
    ]:
        output = model(geometry, time)
        assert output.shape == (2, 3, 100, 100)
        assert torch.count_nonzero(output * (1.0 - geometry[:, :1])) == 0
        output.square().mean().backward()
    proposed_geometry = torch.cat([geometry, torch.rand(2, 2, 100, 100)], dim=1)
    proposed = SteadyResidualHybridUNet(in_channels=5, base=8, bottleneck_modes=2)
    proposed_output = proposed(proposed_geometry, time)
    assert proposed_output.shape == (2, 3, 100, 100)
    assert torch.count_nonzero(proposed_output * (1.0 - geometry[:, :1])) == 0
    proposed_output.square().mean().backward()


if __name__ == "__main__":
    test_output_shape_and_solid_mask()
    print("shape_and_backward_ok")
