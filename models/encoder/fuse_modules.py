from typing import Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
import torch.utils.checkpoint as checkpoint


from models.encoder.vmamba import Mlp, gMlp, Temporal_SSM, SS2D

class SpectralWavePropagator(nn.Module):
    # Spectral Wave Propagator (SWP)
    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.linear = nn.Linear(channels, 2 * channels, bias=True)
        self.out_norm = nn.LayerNorm(channels)
        self.out_linear = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.ReLU(),
        )

        self.c = nn.Parameter(torch.ones(1))
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

    @staticmethod
    def _get_cos_map(n, device, dtype):
        weight_x = (torch.arange(n, device=device, dtype=dtype).view(1, -1) + 0.5) / n
        weight_n = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * math.pi) * math.sqrt(2.0 / n)
        weight[0, :] = weight[0, :] / math.sqrt(2.0)
        return weight

    def _ensure_cos_cache(self, h, w, device, dtype):
        cache = getattr(self, "_cos_cache", None)
        if cache is None or cache["res"] != (h, w) or cache["device"] != device or cache["dtype"] != dtype:
            cosn = self._get_cos_map(h, device, dtype)
            cosm = self._get_cos_map(w, device, dtype)
            cache = {
                "res": (h, w),
                "device": device,
                "dtype": dtype,
                "cosn": cosn,
                "cosm": cosm,
                "cosn_t": cosn.t().contiguous(),
                "cosm_t": cosm.t().contiguous(),
            }
            setattr(self, "_cos_cache", cache)
        return cache

    def _dct2d_nhwc(self, x):
        b, h, w, c = x.shape
        cache = self._ensure_cos_cache(h, w, x.device, x.dtype)
        cosn = cache["cosn"]
        cosm = cache["cosm"]
        cosn_kernel = cosn.view(h, 1, h)
        cosm_kernel = cosm.view(w, 1, w)

        x_perm = x.permute(0, 3, 2, 1).contiguous()
        x_flat_h = x_perm.view(-1, 1, h)
        x_u0 = F.conv1d(x_flat_h, cosn_kernel).squeeze(-1)
        x_u0 = x_u0.view(b, c, w, h).permute(0, 3, 2, 1).contiguous()

        x_perm = x_u0.permute(0, 3, 1, 2).contiguous()
        x_flat_w = x_perm.view(-1, 1, w)
        x_u0 = F.conv1d(x_flat_w, cosm_kernel).squeeze(-1)
        x_u0 = x_u0.view(b, c, h, w).permute(0, 2, 3, 1).contiguous()
        return x_u0

    def _idct2d_nhwc(self, x):
        b, h, w, c = x.shape
        cache = self._ensure_cos_cache(h, w, x.device, x.dtype)
        cosn_t = cache["cosn_t"]
        cosm_t = cache["cosm_t"]
        cosm_kernel_t = cosm_t.contiguous().view(w, 1, w)
        cosn_kernel_t = cosn_t.contiguous().view(h, 1, h)

        x_w = x.permute(0, 1, 3, 2).contiguous().view(b * h * c, 1, w)
        x_w = F.conv1d(x_w, cosm_kernel_t).squeeze(-1)
        x_w = x_w.view(b, h, c, w).permute(0, 1, 3, 2).contiguous()

        x_h = x_w.permute(0, 2, 3, 1).contiguous().view(b * w * c, 1, h)
        x_h = F.conv1d(x_h, cosn_kernel_t).squeeze(-1)
        x_final = x_h.view(b, w, c, h).permute(0, 3, 1, 2).contiguous()
        return x_final

    def forward(self, x: torch.Tensor, freq_embed=None):
        b, c, h, w = x.shape
        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())
        x, z = x.chunk(chunks=2, dim=-1)

        x_u0 = self._dct2d_nhwc(x)
        x_v0 = self._dct2d_nhwc(x)

        if freq_embed is not None:
            if freq_embed.shape[0] != h or freq_embed.shape[1] != w or freq_embed.shape[2] != c:
                raise ValueError(
                    f"freq_embed must be HxWxC = {h}x{w}x{c}, got {tuple(freq_embed.shape)}"
                )
            freq = freq_embed.to(device=x.device, dtype=x.dtype)
            t = self.to_k(freq).unsqueeze(0).expand(b, -1, -1, -1)
        else:
            t = torch.zeros((b, h, w, c), device=x.device, dtype=x.dtype)
        cos_term = torch.cos(self.c * t)
        sin_term = torch.sin(self.c * t) / (self.c + 1e-6)

        wave_term = cos_term * x_u0
        velocity_term = sin_term * (x_v0 + (self.alpha / 2) * x_u0)
        x_final = self._idct2d_nhwc(wave_term + velocity_term)

        x_out = self.out_norm(x_final)
        x_out = x_out * F.silu(z)
        x_out = self.out_linear(x_out)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()
        return x_out


class SpatialRefinementGate(nn.Module):
    # Spatial Refinement Gate (SRG)
    def __init__(self, kernel_size=7):
        super(SpatialRefinementGate, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)    # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        x = torch.cat([avg_out, max_out], dim=1)    # [B, 2, H, W]
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelRefinementGate(nn.Module):
    # Channel Refinement Gate (CRG)
    def __init__(self, in_planes, ratio=16):
        super(ChannelRefinementGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class CCWG(nn.Module): 
    # Cross-Modal Channel-Wave Gating (CCWG)
    def __init__(self, channels, space_dim=None):
        super().__init__()
        self.wave = SpectralWavePropagator(channels)
        self.freq_embed = None
        if space_dim is not None:
            h, w = space_dim
            self.freq_embed = nn.Parameter(torch.zeros(h, w, channels))
            trunc_normal_(self.freq_embed, std=.02)
        self.gate_img = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.gate_ev = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x_img, x_ev):
        img_wave = self.wave(x_img, self.freq_embed)
        ev_wave = self.wave(x_ev, self.freq_embed)
        gate_img = self.gate_img(ev_wave)
        gate_ev = self.gate_ev(img_wave)
        x_img_new = x_img * gate_img
        x_ev_new = x_ev * gate_ev
        return x_img_new, x_ev_new


class CSWG(nn.Module):
    # Cross-Modal Spatial-Wave Gating (CSWG)
    def __init__(self, channels, space_dim=None):
        super().__init__()
        self.wave = SpectralWavePropagator(channels)
        self.freq_embed = None
        if space_dim is not None:
            h, w = space_dim
            self.freq_embed = nn.Parameter(torch.zeros(h, w, channels))
            trunc_normal_(self.freq_embed, std=.02)
        self.gate_img = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.gate_ev = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x_img, x_ev):
        img_wave = self.wave(x_img, self.freq_embed)
        ev_wave = self.wave(x_ev, self.freq_embed)
        gate_img = self.gate_img(ev_wave)
        gate_ev = self.gate_ev(img_wave)
        x_img_new = x_img * gate_img
        x_ev_new = x_ev * gate_ev
        return x_img_new, x_ev_new



class MWCMF(nn.Module): # Mamba-Wave Cross-Modal Fusion MWCMF
    def __init__(
        self,
        hidden_dim: int = 0,
        hidden_space_dim: int = 0,
        space_dim=None,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        # =============================
        mlp_ratio=4.0,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        # =============================
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_space_dim = hidden_space_dim
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.norm_i = norm_layer(hidden_dim)
        self.norm_e = norm_layer(hidden_dim)

        # ========================= CTIM (WaveFormer) ===========================
        self.cca = CCWG(hidden_dim, space_dim=space_dim)

        self.ssm_i = Temporal_SSM(
            d_model=hidden_dim,
            d_ssm=hidden_space_dim,
            d_state=ssm_d_state,
            ssm_ratio=2.0,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="vc_noz",
            channel_first=channel_first,
            **kwargs,
        )
        self.ssm_e = Temporal_SSM(
            d_model=hidden_dim,
            d_ssm=hidden_space_dim,
            d_state=ssm_d_state,
            ssm_ratio=2.0,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="vc_noz",
            channel_first=channel_first,
            **kwargs,
        )

        self.conv_i1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        self.conv_e1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)

        self.ca1 = ChannelRefinementGate(hidden_dim)
        self.ca2 = ChannelRefinementGate(hidden_dim)

        # ========================= CSIM (WaveFormer) ===========================
        self.csa = CSWG(hidden_dim, space_dim=space_dim)

        self.ss2d_i = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="v05_noz",
            channel_first=channel_first,
        )
        self.ss2d_e = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="v05_noz",
            channel_first=channel_first,
        )

        self.conv_i2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        self.conv_e2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)

        self.sa1 = SpatialRefinementGate()
        self.sa2 = SpatialRefinementGate()

        self.conv_i3 = nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=1)
        self.conv_e3 = nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=1)

        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

    def _forward(self, x_img: torch.Tensor, x_ev: torch.Tensor):
        img_norm = self.norm_i(x_img)
        ev_norm = self.norm_e(x_ev)

        img_cca, ev_cca = self.cca(img_norm, ev_norm)
        img_csa, ev_csa = self.csa(img_norm, ev_norm)

        x_img_c = self.ssm_i(img_cca) + self.conv_i1(img_cca)
        x_ev_c = self.ssm_e(ev_cca) + self.conv_e1(ev_cca)

        x_img_c = x_img_c * self.ca1(x_img_c)
        x_ev_c = x_ev_c * self.ca2(x_ev_c)

        x_img_s = self.ss2d_i(img_csa) + self.conv_i2(img_csa)
        x_ev_s = self.ss2d_e(ev_csa) + self.conv_e2(ev_csa)

        x_img_s = x_img_s * self.sa1(x_img_s)
        x_ev_s = x_ev_s * self.sa2(x_ev_s)

        x_img_ = torch.cat((x_img_c, x_img_s), dim=1)
        x_ev_ = torch.cat((x_ev_c, x_ev_s), dim=1)
        x_img_swap = self.conv_i3(x_img_)
        x_ev_swap = self.conv_e3(x_ev_)
        x_img = self.drop_path1(x_img_swap) + x_img
        x_ev = self.drop_path2(x_ev_swap) + x_ev

        return x_img, x_ev

    def forward(self, x_img: torch.Tensor, x_ev: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_img, x_ev)
        else:
            return self._forward(x_img, x_ev)
        



class MWCMF_MambaWaveFast(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        hidden_space_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        # =============================
        mlp_ratio=4.0,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        # =============================
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_space_dim = hidden_space_dim
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.norm_i = norm_layer(hidden_dim)
        self.norm_e = norm_layer(hidden_dim)

        # ========================= CTIM (Fast Wave + Mamba) ===========================
        self.cca = CrossTemporalMambaWaveFast(
            hidden_dim,
            hidden_space_dim=hidden_space_dim,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            ssm_init=ssm_init,
            channel_first=channel_first,
            **kwargs,
        )

        self.ssm_i = Temporal_SSM(
            d_model=hidden_dim,
            d_ssm=hidden_space_dim,
            d_state=ssm_d_state,
            ssm_ratio=2.0,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="vc_noz",
            channel_first=channel_first,
            **kwargs,
        )
        self.ssm_e = Temporal_SSM(
            d_model=hidden_dim,
            d_ssm=hidden_space_dim,
            d_state=ssm_d_state,
            ssm_ratio=2.0,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="vc_noz",
            channel_first=channel_first,
            **kwargs,
        )

        self.conv_i1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        self.conv_e1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)

        self.ca1 = ChannelRefinementGate(hidden_dim)
        self.ca2 = ChannelRefinementGate(hidden_dim)

        # ========================= CSIM (Fast Wave + Mamba) ===========================
        self.csa = CrossSpatialMambaWaveFast(
            hidden_dim,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            ssm_init=ssm_init,
            channel_first=channel_first,
            **kwargs,
        )

        self.ss2d_i = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="v05_noz",
            channel_first=channel_first,
        )
        self.ss2d_e = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="v05_noz",
            channel_first=channel_first,
        )

        self.conv_i2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        self.conv_e2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)

        self.sa1 = SpatialRefinementGate()
        self.sa2 = SpatialRefinementGate()

        self.conv_i3 = nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=1)
        self.conv_e3 = nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=1)

        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

    def _forward(self, x_img: torch.Tensor, x_ev: torch.Tensor):
        img_norm = self.norm_i(x_img)
        ev_norm = self.norm_e(x_ev)

        img_cca, ev_cca = self.cca(img_norm, ev_norm)
        img_csa, ev_csa = self.csa(img_norm, ev_norm)

        x_img_c = self.ssm_i(img_cca) + self.conv_i1(img_cca)
        x_ev_c = self.ssm_e(ev_cca) + self.conv_e1(ev_cca)

        x_img_c = x_img_c * self.ca1(x_img_c)
        x_ev_c = x_ev_c * self.ca2(x_ev_c)

        x_img_s = self.ss2d_i(img_csa) + self.conv_i2(img_csa)
        x_ev_s = self.ss2d_e(ev_csa) + self.conv_e2(ev_csa)

        x_img_s = x_img_s * self.sa1(x_img_s)
        x_ev_s = x_ev_s * self.sa2(x_ev_s)

        x_img_ = torch.cat((x_img_c, x_img_s), dim=1)
        x_ev_ = torch.cat((x_ev_c, x_ev_s), dim=1)
        x_img_swap = self.conv_i3(x_img_)
        x_ev_swap = self.conv_e3(x_ev_)
        x_img = self.drop_path1(x_img_swap) + x_img
        x_ev = self.drop_path2(x_ev_swap) + x_ev

        return x_img, x_ev

    def forward(self, x_img: torch.Tensor, x_ev: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_img, x_ev)
        else:
            return self._forward(x_img, x_ev)


class CrossSpatialMambaWaveFast(nn.Module):
    def __init__(
        self,
        channels,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        channel_first=False,
        **kwargs,
    ):
        super().__init__()
        self.ss2d_img = SS2D(
            d_model=channels,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="v05_noz",
            channel_first=channel_first,
            **kwargs,
        )
        self.ss2d_ev = SS2D(
            d_model=channels,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="v05_noz",
            channel_first=channel_first,
            **kwargs,
        )
        self.wave = WavePropagation2DFast(channels)
        self.gate_img = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.gate_ev = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x_img, x_ev):
        img_feat = self.ss2d_img(x_img)
        ev_feat = self.ss2d_ev(x_ev)
        img_wave = self.wave(img_feat)
        ev_wave = self.wave(ev_feat)
        gate_img = self.gate_img(ev_wave)
        gate_ev = self.gate_ev(img_wave)
        x_img_new = x_img * gate_img
        x_ev_new = x_ev * gate_ev
        return x_img_new, x_ev_new



class DDIM_WaveFast(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        hidden_space_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        # =============================
        mlp_ratio=4.0,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        # =============================
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_space_dim = hidden_space_dim
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.norm_i = norm_layer(hidden_dim)
        self.norm_e = norm_layer(hidden_dim)

        # ========================= CTIM (Fast Wave) ===========================
        self.cca = CrossTemporalWaveFast(hidden_dim)

        self.ssm_i = Temporal_SSM(
            d_model=hidden_dim,
            d_ssm=hidden_space_dim,
            d_state=ssm_d_state,
            ssm_ratio=2.0,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="vc_noz",
            channel_first=channel_first,
            **kwargs,
        )
        self.ssm_e = Temporal_SSM(
            d_model=hidden_dim,
            d_ssm=hidden_space_dim,
            d_state=ssm_d_state,
            ssm_ratio=2.0,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="vc_noz",
            channel_first=channel_first,
            **kwargs,
        )

        self.conv_i1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        self.conv_e1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)

        self.ca1 = ChannelRefinementGate(hidden_dim)
        self.ca2 = ChannelRefinementGate(hidden_dim)

        # ========================= CSIM (Fast Wave) ===========================
        self.csa = CrossSpatialWaveFast(hidden_dim)

        self.ss2d_i = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="v05_noz",
            channel_first=channel_first,
        )
        self.ss2d_e = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="v05_noz",
            channel_first=channel_first,
        )

        self.conv_i2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        self.conv_e2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)

        self.sa1 = SpatialRefinementGate()
        self.sa2 = SpatialRefinementGate()

        self.conv_i3 = nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=1)
        self.conv_e3 = nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=1)

        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

    def _forward(self, x_img: torch.Tensor, x_ev: torch.Tensor):
        img_norm = self.norm_i(x_img)
        ev_norm = self.norm_e(x_ev)

        img_cca, ev_cca = self.cca(img_norm, ev_norm)
        img_csa, ev_csa = self.csa(img_norm, ev_norm)

        x_img_c = self.ssm_i(img_cca) + self.conv_i1(img_cca)
        x_ev_c = self.ssm_e(ev_cca) + self.conv_e1(ev_cca)

        x_img_c = x_img_c * self.ca1(x_img_c)
        x_ev_c = x_ev_c * self.ca2(x_ev_c)

        x_img_s = self.ss2d_i(img_csa) + self.conv_i2(img_csa)
        x_ev_s = self.ss2d_e(ev_csa) + self.conv_e2(ev_csa)

        x_img_s = x_img_s * self.sa1(x_img_s)
        x_ev_s = x_ev_s * self.sa2(x_ev_s)

        x_img_ = torch.cat((x_img_c, x_img_s), dim=1)
        x_ev_ = torch.cat((x_ev_c, x_ev_s), dim=1)
        x_img_swap = self.conv_i3(x_img_)
        x_ev_swap = self.conv_e3(x_ev_)
        x_img = self.drop_path1(x_img_swap) + x_img
        x_ev = self.drop_path2(x_ev_swap) + x_ev

        return x_img, x_ev

    def forward(self, x_img: torch.Tensor, x_ev: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_img, x_ev)
        else:
            return self._forward(x_img, x_ev)


class CrossSpatialWaveFast(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.wave = WavePropagation2DFast(channels)
        self.gate_img = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.gate_ev = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x_img, x_ev):
        img_wave = self.wave(x_img)
        ev_wave = self.wave(x_ev)
        gate_img = self.gate_img(ev_wave)
        gate_ev = self.gate_ev(img_wave)
        x_img_new = x_img * gate_img
        x_ev_new = x_ev * gate_ev
        return x_img_new, x_ev_new



class CrossTemporalMambaWaveFast(nn.Module):
    def __init__(
        self,
        channels,
        hidden_space_dim,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        channel_first=False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_img = Temporal_SSM(
            d_model=channels,
            d_ssm=hidden_space_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="vc_noz",
            channel_first=channel_first,
            **kwargs,
        )
        self.ssm_ev = Temporal_SSM(
            d_model=channels,
            d_ssm=hidden_space_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="vc_noz",
            channel_first=channel_first,
            **kwargs,
        )
        self.wave = WavePropagation2DFast(channels)
        self.gate_img = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.gate_ev = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x_img, x_ev):
        img_feat = self.ssm_img(x_img)
        ev_feat = self.ssm_ev(x_ev)
        img_wave = self.wave(img_feat)
        ev_wave = self.wave(ev_feat)
        gate_img = self.gate_img(ev_wave)
        gate_ev = self.gate_ev(img_wave)
        x_img_new = x_img * gate_img
        x_ev_new = x_ev * gate_ev
        return x_img_new, x_ev_new

class WavePropagation2DFast(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels, 1, 1))
        self.omega = nn.Parameter(torch.ones(channels, 1, 1))

    @staticmethod
    def _make_k_cache(h, w, device, dtype):
        kx = torch.fft.fftfreq(h, device=device, dtype=dtype)
        ky = torch.fft.rfftfreq(w, device=device, dtype=dtype)
        k2 = (kx[:, None] ** 2 + ky[None, :] ** 2).unsqueeze(0).unsqueeze(0)
        k = torch.sqrt(k2 + 1e-6)
        return {"k2": k2, "k": k, "res": (h, w), "device": device, "dtype": dtype}

    def _ensure_k_cache(self, h, w, device, dtype):
        cache = getattr(self, "_k_cache", None)
        if cache is None or cache["res"] != (h, w) or cache["device"] != device or cache["dtype"] != dtype:
            cache = self._make_k_cache(h, w, device, dtype)
            setattr(self, "_k_cache", cache)
        return cache

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        cache = self._ensure_k_cache(h, w, x.device, x.dtype)
        k2 = cache["k2"]
        k = cache["k"]

        alpha = F.softplus(self.alpha).unsqueeze(0)
        omega = self.omega.unsqueeze(0)
        filt = torch.exp(-alpha * k2) * torch.cos(omega * k)

        x_f = torch.fft.rfft2(x, dim=(-2, -1))
        y_f = x_f * filt
        y = torch.fft.irfft2(y_f, s=(h, w), dim=(-2, -1))
        return y




class CrossTemporalWaveFast(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.wave = WavePropagation2DFast(channels)
        self.gate_img = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.gate_ev = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x_img, x_ev):
        img_wave = self.wave(x_img)
        ev_wave = self.wave(x_ev)
        gate_img = self.gate_img(ev_wave)
        gate_ev = self.gate_ev(img_wave)
        x_img_new = x_img * gate_img
        x_ev_new = x_ev * gate_ev
        return x_img_new, x_ev_new

