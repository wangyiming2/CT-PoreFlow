from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("time embedding dimension must be even")
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            torch.linspace(0, -math.log(10000.0), half, device=time.device, dtype=time.dtype)
        )
        angles = time[:, None] * frequencies[None, :] * 2.0 * math.pi
        return torch.cat([angles.sin(), angles.cos()], dim=1)


class TimeResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.time_affine = nn.Linear(time_dim, 2 * out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.conv1(x)))
        scale, shift = self.time_affine(time_embedding).chunk(2, dim=1)
        hidden = self.norm2(self.conv2(hidden))
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return F.silu(hidden + self.skip(x))


class TimeConditionedUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base: int = 32) -> None:
        super().__init__()
        time_dim = base * 4
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.enc1 = TimeResBlock(in_channels, base, time_dim)
        self.enc2 = TimeResBlock(base, base * 2, time_dim)
        self.enc3 = TimeResBlock(base * 2, base * 4, time_dim)
        self.enc4 = TimeResBlock(base * 4, base * 8, time_dim)
        self.mid = TimeResBlock(base * 8, base * 16, time_dim)
        self.dec4 = TimeResBlock(base * 16 + base * 8, base * 8, time_dim)
        self.dec3 = TimeResBlock(base * 8 + base * 4, base * 4, time_dim)
        self.dec2 = TimeResBlock(base * 4 + base * 2, base * 2, time_dim)
        self.dec1 = TimeResBlock(base * 2 + base, base, time_dim)
        self.out = nn.Conv2d(base, out_channels, 1)

    @staticmethod
    def down(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=2, ceil_mode=False)

    @staticmethod
    def up_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, geometry: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        embedding = self.time_embedding(time)
        e1 = self.enc1(geometry, embedding)
        e2 = self.enc2(self.down(e1), embedding)
        e3 = self.enc3(self.down(e2), embedding)
        e4 = self.enc4(self.down(e3), embedding)
        hidden = self.mid(self.down(e4), embedding)
        hidden = self.dec4(torch.cat([self.up_like(hidden, e4), e4], dim=1), embedding)
        hidden = self.dec3(torch.cat([self.up_like(hidden, e3), e3], dim=1), embedding)
        hidden = self.dec2(torch.cat([self.up_like(hidden, e2), e2], dim=1), embedding)
        hidden = self.dec1(torch.cat([self.up_like(hidden, e1), e1], dim=1), embedding)
        pore = geometry[:, :1]
        return self.out(hidden) * pore


class GeometryResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.main(x) + self.skip(x))


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight_top = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, 2)
        )
        self.weight_bottom = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, 2)
        )

    @staticmethod
    def multiply(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, torch.view_as_complex(weights))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        input_dtype = x.dtype
        # cuFFT does not support arbitrary (e.g. 100 x 100) signal sizes in
        # float16. The spectral operation is intentionally evaluated in FP32
        # even when the surrounding network uses AMP, then cast back.
        with torch.autocast(device_type=x.device.type, enabled=False):
            transformed = torch.fft.rfft2(x.float(), norm="ortho")
            modes_x = min(self.modes_x, height // 2)
            modes_y = min(self.modes_y, width // 2 + 1)
            output = torch.zeros(
                batch,
                self.out_channels,
                height,
                width // 2 + 1,
                device=x.device,
                dtype=transformed.dtype,
            )
            output[:, :, :modes_x, :modes_y] = self.multiply(
                transformed[:, :, :modes_x, :modes_y],
                self.weight_top[:, :, :modes_x, :modes_y].float(),
            )
            output[:, :, -modes_x:, :modes_y] = self.multiply(
                transformed[:, :, -modes_x:, :modes_y],
                self.weight_bottom[:, :, :modes_x, :modes_y].float(),
            )
            result = torch.fft.irfft2(output, s=(height, width), norm="ortho")
        return result.to(dtype=input_dtype)


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.local = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(min(8, width), width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.local(x)))


class TimeConditionedFNO(nn.Module):
    """2-D FNO queried at a continuous log-time value."""

    def __init__(self, in_channels: int = 3, width: int = 48, modes: int = 16, layers: int = 4) -> None:
        super().__init__()
        # pore, connected pore, EDT, time, x coordinate, y coordinate
        self.lift = nn.Conv2d(in_channels + 3, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes, modes) for _ in range(layers)])
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv2d(width * 2, 3, 1),
        )

    def forward(self, geometry: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = geometry.shape
        x_coord = torch.linspace(0.0, 1.0, height, device=geometry.device, dtype=geometry.dtype)
        y_coord = torch.linspace(0.0, 1.0, width, device=geometry.device, dtype=geometry.dtype)
        x_map = x_coord[None, None, :, None].expand(batch, 1, height, width)
        y_map = y_coord[None, None, None, :].expand(batch, 1, height, width)
        time_map = time[:, None, None, None].expand(batch, 1, height, width)
        hidden = self.lift(torch.cat([geometry, time_map, x_map, y_map], dim=1))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.project(hidden) * geometry[:, :1]


class GeometryDeepONet(nn.Module):
    """Vanilla branch-trunk operator baseline on the regular pore grid."""

    def __init__(self, in_channels: int = 3, basis: int = 64, branch_width: int = 32, trunk_width: int = 128) -> None:
        super().__init__()
        self.basis = basis
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, branch_width, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(branch_width, branch_width * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(branch_width * 2, branch_width * 4, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(branch_width * 4, branch_width * 8, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(branch_width * 8, 3 * basis),
        )
        self.trunk = nn.Sequential(
            nn.Linear(3, trunk_width),
            nn.GELU(),
            nn.Linear(trunk_width, trunk_width),
            nn.GELU(),
            nn.Linear(trunk_width, trunk_width),
            nn.GELU(),
            nn.Linear(trunk_width, basis),
        )
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, geometry: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = geometry.shape
        branch = self.branch(geometry).view(batch, 3, self.basis)
        x = torch.linspace(0.0, 1.0, height, device=geometry.device, dtype=geometry.dtype)
        y = torch.linspace(0.0, 1.0, width, device=geometry.device, dtype=geometry.dtype)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        coordinates = torch.stack(
            [
                grid_x[None].expand(batch, height, width),
                grid_y[None].expand(batch, height, width),
                time[:, None, None].expand(batch, height, width),
            ],
            dim=-1,
        )
        trunk = self.trunk(coordinates.reshape(batch * height * width, 3)).view(
            batch, height * width, self.basis
        )
        output = torch.einsum("bcp,bnp->bcn", branch, trunk)
        output = output.view(batch, 3, height, width) + self.bias[None, :, None, None]
        return output * geometry[:, :1]


class SteadyResidualHybridUNet(nn.Module):
    """Topology-aware U-Net with a spectral bottleneck and steady/transient decomposition."""

    def __init__(
        self,
        in_channels: int = 5,
        base: int = 32,
        bottleneck_modes: int = 4,
        use_spectral_bottleneck: bool = True,
        use_steady_anchor: bool = True,
    ) -> None:
        super().__init__()
        self.use_steady_anchor = use_steady_anchor
        time_dim = base * 4
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.enc1 = GeometryResBlock(in_channels, base)
        self.enc2 = GeometryResBlock(base, base * 2)
        self.enc3 = GeometryResBlock(base * 2, base * 4)
        self.enc4 = GeometryResBlock(base * 4, base * 8)
        self.mid = GeometryResBlock(base * 8, base * 16)
        self.spectral_mid = (
            FNOBlock(base * 16, bottleneck_modes, bottleneck_modes)
            if use_spectral_bottleneck
            else None
        )

        if use_steady_anchor:
            self.steady4 = GeometryResBlock(base * 16 + base * 8, base * 8)
            self.steady3 = GeometryResBlock(base * 8 + base * 4, base * 4)
            self.steady2 = GeometryResBlock(base * 4 + base * 2, base * 2)
            self.steady1 = GeometryResBlock(base * 2 + base, base)
            self.steady_out = nn.Conv2d(base, 3, 1)

        self.transient4 = TimeResBlock(base * 16 + base * 8, base * 8, time_dim)
        self.transient3 = TimeResBlock(base * 8 + base * 4, base * 4, time_dim)
        self.transient2 = TimeResBlock(base * 4 + base * 2, base * 2, time_dim)
        self.transient1 = TimeResBlock(base * 2 + base, base, time_dim)
        self.transient_out = nn.Conv2d(base, 3, 1)

    @staticmethod
    def down(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, 2)

    @staticmethod
    def up_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, geometry: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(geometry)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        e4 = self.enc4(self.down(e3))
        mid = self.mid(self.down(e4))
        if self.spectral_mid is not None:
            mid = mid + self.spectral_mid(mid)

        embedding = self.time_embedding(time)
        transient = self.transient4(torch.cat([self.up_like(mid, e4), e4], dim=1), embedding)
        transient = self.transient3(torch.cat([self.up_like(transient, e3), e3], dim=1), embedding)
        transient = self.transient2(torch.cat([self.up_like(transient, e2), e2], dim=1), embedding)
        transient = self.transient1(torch.cat([self.up_like(transient, e1), e1], dim=1), embedding)
        transient = self.transient_out(transient)

        if not self.use_steady_anchor:
            return transient * geometry[:, :1]

        steady = self.steady4(torch.cat([self.up_like(mid, e4), e4], dim=1))
        steady = self.steady3(torch.cat([self.up_like(steady, e3), e3], dim=1))
        steady = self.steady2(torch.cat([self.up_like(steady, e2), e2], dim=1))
        steady = self.steady1(torch.cat([self.up_like(steady, e1), e1], dim=1))
        steady = self.steady_out(steady)
        decay = (1.0 - time).clamp(0.0, 1.0)[:, None, None, None]
        return (steady + decay * transient) * geometry[:, :1]


class ConvLSTMCell2d(nn.Module):
    """Convolutional LSTM cell used as a learned latent-time integrator."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat([inputs, hidden], dim=1)
        ).chunk(4, dim=1)
        input_gate = input_gate.sigmoid()
        forget_gate = forget_gate.sigmoid()
        output_gate = output_gate.sigmoid()
        candidate = candidate.tanh()
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * cell.tanh()
        return hidden, cell


class QueryTimeConvLSTM(nn.Module):
    """Geometry encoder plus recurrent latent evolution to a continuous query time.

    The complete recurrent path is traversed for every supervision time.  The
    per-step ``dt`` is the normalized query time divided by the number of latent
    integration steps, so the architecture can be trained with the existing
    randomly queried frames while still learning an explicit state evolution.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base: int = 24,
        hidden_channels: int = 64,
        recurrent_steps: int = 8,
    ) -> None:
        super().__init__()
        self.recurrent_steps = recurrent_steps
        self.encoder = nn.Sequential(
            GeometryResBlock(in_channels, base),
            nn.AvgPool2d(2),
            GeometryResBlock(base, base * 2),
            nn.AvgPool2d(2),
            GeometryResBlock(base * 2, hidden_channels),
        )
        self.initial_hidden = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.initial_cell = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.cell = ConvLSTMCell2d(hidden_channels + 1, hidden_channels)
        self.decoder = nn.Sequential(
            GeometryResBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, base * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.SiLU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(base + in_channels, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base, 3, 1),
        )

    def forward(self, geometry: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        context = self.encoder(geometry)
        hidden = self.initial_hidden(context).tanh()
        cell = self.initial_cell(context).tanh()
        dt = (time / float(self.recurrent_steps))[:, None, None, None]
        dt_map = dt.expand(-1, 1, context.shape[-2], context.shape[-1])
        recurrent_input = torch.cat([context, dt_map], dim=1)
        for _ in range(self.recurrent_steps):
            hidden, cell = self.cell(recurrent_input, hidden, cell)
        decoded = self.decoder(hidden)
        decoded = F.interpolate(
            decoded, size=geometry.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.refine(torch.cat([decoded, geometry], dim=1)) * geometry[:, :1]


class GeometryTimeTransformer(nn.Module):
    """Patch-token Transformer with a dedicated continuous-time query token."""

    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 96,
        heads: int = 4,
        layers: int = 4,
        patch_size: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("transformer d_model must be divisible by heads")
        self.patch_size = patch_size
        self.patch = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.position = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model // 4, 3, padding=1),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(d_model // 4 + in_channels, d_model // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model // 4, 3, 1),
        )

    def forward(self, geometry: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        patch_map = self.patch(geometry)
        batch, channels, height, width = patch_map.shape
        x = torch.linspace(0.0, 1.0, height, device=geometry.device, dtype=geometry.dtype)
        y = torch.linspace(0.0, 1.0, width, device=geometry.device, dtype=geometry.dtype)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        coordinates = torch.stack([grid_x, grid_y], dim=-1).reshape(height * width, 2)
        position = self.position(coordinates)[None]
        tokens = patch_map.flatten(2).transpose(1, 2) + position
        time_token = self.time_embedding(time)[:, None, :]
        tokens = torch.cat([time_token, tokens], dim=1)
        encoded = self.norm(self.transformer(tokens))
        # The time token is broadcast back to the spatial tokens after global
        # self-attention, retaining an explicit query-time conditioning path.
        spatial = encoded[:, 1:] + encoded[:, :1]
        spatial = spatial.transpose(1, 2).reshape(batch, channels, height, width)
        decoded = self.decoder(spatial)
        decoded = F.interpolate(
            decoded, size=geometry.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.refine(torch.cat([decoded, geometry], dim=1)) * geometry[:, :1]


def build_model(config: dict) -> nn.Module:
    name = config.get("model", "time_unet")
    in_channels = int(config.get("input_channels", 3))
    if name == "time_unet":
        return TimeConditionedUNet(in_channels=in_channels, base=int(config.get("base_channels", 32)))
    if name == "time_fno":
        return TimeConditionedFNO(
            in_channels=in_channels,
            width=int(config.get("fno_width", 48)),
            modes=int(config.get("fno_modes", 16)),
            layers=int(config.get("fno_layers", 4)),
        )
    if name == "deeponet":
        return GeometryDeepONet(
            in_channels=in_channels,
            basis=int(config.get("deeponet_basis", 64)),
            branch_width=int(config.get("deeponet_branch_width", 32)),
            trunk_width=int(config.get("deeponet_trunk_width", 128)),
        )
    if name == "steady_residual_hybrid":
        return SteadyResidualHybridUNet(
            in_channels=in_channels,
            base=int(config.get("base_channels", 32)),
            bottleneck_modes=int(config.get("bottleneck_modes", 4)),
            use_spectral_bottleneck=bool(config.get("use_spectral_bottleneck", True)),
            use_steady_anchor=bool(config.get("use_steady_anchor", True)),
        )
    if name == "query_time_convlstm":
        return QueryTimeConvLSTM(
            in_channels=in_channels,
            base=int(config.get("convlstm_base_channels", 24)),
            hidden_channels=int(config.get("convlstm_hidden_channels", 64)),
            recurrent_steps=int(config.get("convlstm_recurrent_steps", 8)),
        )
    if name == "geometry_time_transformer":
        return GeometryTimeTransformer(
            in_channels=in_channels,
            d_model=int(config.get("transformer_d_model", 96)),
            heads=int(config.get("transformer_heads", 4)),
            layers=int(config.get("transformer_layers", 4)),
            patch_size=int(config.get("transformer_patch_size", 5)),
            dropout=float(config.get("transformer_dropout", 0.0)),
        )
    raise ValueError(f"Unknown model: {name}")
