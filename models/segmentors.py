from math import ceil
from os.path import exists
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder.fuse_modules import (
    MWCMF,
    MWCMF_MambaWaveFast,
    DDIM_WaveFast,
)
from models.encoder.groupmamba import Stem, DownSamples, Block_mamba


class StageBlock(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        raise NotImplementedError("Use .blocks directly with token input and spatial size.")


class DownsampleBlock(nn.Module):
    def __init__(self, downsample):
        super().__init__()
        self.downsample = downsample


class GroupMambaStage(nn.Module):
    def __init__(self, blocks, downsample=None):
        super().__init__()
        self.blocks = blocks
        self.downsample = downsample if downsample is not None else nn.Identity()


class Backbone_GroupMamba(nn.Module):
    """
    Segmentation backbone wrapper for GroupMamba.
    It exposes the same stage-wise interface MambaSeg expects:
      - patch_embed(x) -> tokens or NCHW stage-0 input
      - layers[i].blocks(x) -> stage-i output in NCHW
      - layers[i].downsample(x) -> next-stage token input
      - outnorm{i}(x) -> normalized stage feature map
    """
    ARCH_SETTINGS = {
        'groupmamba_tiny': {
            'stem_hidden_dim': 32,
            'dims': [64, 128, 348, 448],
            'mlp_ratios': [8, 8, 4, 4],
            'depths': [3, 4, 9, 3],
        },
        'groupmamba_small': {
            'stem_hidden_dim': 64,
            'dims': [64, 128, 348, 512],
            'mlp_ratios': [8, 8, 4, 4],
            'depths': [3, 4, 16, 3],
        },
        'groupmamba_base': {
            'stem_hidden_dim': 64,
            'dims': [96, 192, 424, 512],
            'mlp_ratios': [8, 8, 4, 4],
            'depths': [3, 6, 21, 3],
        },
    }

    def __init__(self, version='groupmamba_small', in_chans=3, out_indices=(0, 1, 2, 3), pretrained=None, drop_path_rate=0.2):
        super().__init__()
        if version not in self.ARCH_SETTINGS:
            raise ValueError(f"Unsupported GroupMamba version: {version}")

        cfg = self.ARCH_SETTINGS[version]
        dims = cfg['dims']
        depths = cfg['depths']
        mlp_ratios = cfg['mlp_ratios']
        self.dims = dims
        self.out_indices = out_indices
        self.num_stages = len(dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dp_idx = 0

        self.patch_embed = Stem(
            in_channels=in_chans,
            stem_hidden_dim=cfg['stem_hidden_dim'],
            out_channels=dims[0]
        )

        self.layers = nn.ModuleList()
        for i in range(self.num_stages):
            blocks = nn.ModuleList([
                Block_mamba(
                    dim=dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop_path=dpr[dp_idx + j],
                    norm_layer=nn.LayerNorm,
                )
                for j in range(depths[i])
            ])
            dp_idx += depths[i]

            downsample = None
            if i < self.num_stages - 1:
                downsample = self._make_downsample(dims[i], dims[i + 1])

            self.layers.append(GroupMambaStage(blocks=blocks, downsample=downsample))
            setattr(self, f'outnorm{i}', nn.Identity())

        if pretrained is not None:
            self.load_pretrained(pretrained, in_chans=in_chans)

    @staticmethod
    def _make_downsample(in_dim, out_dim):
        class _Downsample(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.op = DownSamples(in_dim, out_dim)

            def forward(self, x):
                # x: [B, C, H, W] -> tokens [B, HW, C_next]
                return self.op(x)[0]

        return _Downsample(in_dim, out_dim)

    @staticmethod
    def _adapt_input_conv(weight, target_in_chans):
        # weight shape: [out_c, in_c, k, k]
        src_in = weight.shape[1]
        if src_in == target_in_chans:
            return weight
        if target_in_chans == 1:
            return weight.mean(dim=1, keepdim=True)
        if target_in_chans < src_in:
            return weight[:, :target_in_chans, :, :]

        repeat = (target_in_chans + src_in - 1) // src_in
        adapted = weight.repeat(1, repeat, 1, 1)[:, :target_in_chans, :, :]
        adapted = adapted * (src_in / float(target_in_chans))
        return adapted

    def load_pretrained(self, pretrained, in_chans=3):
        if not exists(pretrained):
            print(f"[Backbone_GroupMamba] pretrained file not found: {pretrained}")
            return

        ckpt = torch.load(pretrained, map_location='cpu')
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))

        def strip_prefixes(key):
            for prefix in ('module.', 'backbone.', 'encoder.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            return key

        def remap_key(key):
            key = strip_prefixes(key)

            if key.startswith(('head.', 'dist_head.', 'post_network.', 'norm1', 'norm2', 'norm3', 'norm4')):
                return None

            if key.startswith('patch_embed1.'):
                return 'patch_embed.' + key[len('patch_embed1.'):]
            if key.startswith('patch_embed2.'):
                return 'layers.0.downsample.op.' + key[len('patch_embed2.'):]
            if key.startswith('patch_embed3.'):
                return 'layers.1.downsample.op.' + key[len('patch_embed3.'):]
            if key.startswith('patch_embed4.'):
                return 'layers.2.downsample.op.' + key[len('patch_embed4.'):]

            if key.startswith('block1.'):
                return 'layers.0.blocks.' + key[len('block1.'):]
            if key.startswith('block2.'):
                return 'layers.1.blocks.' + key[len('block2.'):]
            if key.startswith('block3.'):
                return 'layers.2.blocks.' + key[len('block3.'):]
            if key.startswith('block4.'):
                return 'layers.3.blocks.' + key[len('block4.'):]

            if key.startswith('patch_embed.') or key.startswith('layers.') or key.startswith('outnorm'):
                return key

            return None

        remapped = {}
        skipped = []
        for k, v in state.items():
            nk = remap_key(k)
            if nk is None:
                skipped.append(k)
                continue
            remapped[nk] = v

        stem_key = 'patch_embed.conv.0.weight'
        if stem_key in remapped:
            remapped[stem_key] = self._adapt_input_conv(remapped[stem_key], in_chans)

        model_state = self.state_dict()
        filtered = {}
        shape_mismatch = []
        for k, v in remapped.items():
            if k not in model_state:
                skipped.append(k)
                continue
            if model_state[k].shape != v.shape:
                shape_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                continue
            filtered[k] = v

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(f"[Backbone_GroupMamba] loaded {pretrained}")
        print(f"[Backbone_GroupMamba] matched keys: {len(filtered)}/{len(model_state)}")
        if len(shape_mismatch) > 0:
            print(f"[Backbone_GroupMamba] shape mismatches: {len(shape_mismatch)}")
            for name, src_shape, dst_shape in shape_mismatch[:10]:
                print(f"  - {name}: ckpt{src_shape} != model{dst_shape}")
        if len(missing) > 0:
            print(f"[Backbone_GroupMamba] missing keys: {len(missing)}")
        if len(unexpected) > 0:
            print(f"[Backbone_GroupMamba] unexpected keys after remap: {len(unexpected)}")
        if len(skipped) > 0:
            print(f"[Backbone_GroupMamba] skipped keys: {len(skipped)}")


class GroupMambaTokenBlock(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        raise NotImplementedError


class WaveMamba(nn.Module):
    def __init__(self, ver_img='groupmamba_small', ver_ev='groupmamba_small', num_classes=6,
                 fuse=None, num_channels_img=3, pretrained_img=None,
                 num_channels_ev=3, pretrained_ev=None, data_type=None, img_size=None, if_viz=False):
        super().__init__()
        self.num_channels_img = num_channels_img
        self.num_channels_ev = num_channels_ev
        self.out_indices = (0, 1, 2, 3)

        dim_dict = {
            'groupmamba_tiny': [64, 128, 348, 448],
            'groupmamba_small': [64, 128, 348, 512],
            'groupmamba_base': [96, 192, 424, 512],
        }

        if ver_img not in dim_dict or ver_ev not in dim_dict:
            raise ValueError(
                f"Use GroupMamba versions only: {list(dim_dict.keys())}. "
                f"Got ver_img={ver_img}, ver_ev={ver_ev}"
            )

        dim_img = dim_dict[ver_img]
        dim_ev = dim_dict[ver_ev]
        if dim_img != dim_ev:
            raise ValueError("For the current fusion + SegFormer decoder, image/event dims must match.")

        self.patch_size = 4
        space_dim = [(ceil(img_size[0] / self.patch_size), ceil(img_size[1] / self.patch_size))]
        for i in range(len(dim_img) - 1):
            space_dim.append((ceil(space_dim[i][0] / 2), ceil(space_dim[i][1] / 2)))
        self.space_dim = space_dim
        self.num_tokens = [h * w for h, w in self.space_dim]

        self.encoder_img = Backbone_GroupMamba(
            version=ver_img,
            in_chans=num_channels_img,
            out_indices=self.out_indices,
            pretrained=pretrained_img,
            drop_path_rate=0.2,
        )
        self.encoder_ev = Backbone_GroupMamba(
            version=ver_ev,
            in_chans=num_channels_ev,
            out_indices=self.out_indices,
            pretrained=pretrained_ev,
            drop_path_rate=0.2,
        )

        fuse_name = (fuse or "MWCMF").lower()
        if fuse_name in ("MWCMF", "ddim_fuse"):
            fuse_cls = MWCMF
        elif fuse_name in ("ddim_mamba", "mamba", "mamba_fuse"):
            fuse_cls = MWCMF_MambaWaveFast
        elif fuse_name in ("ddim_wave", "wave"):
            fuse_cls = DDIM_WaveFast


        self.ccbs = nn.ModuleList([
            MWCMF(
                hidden_dim=dim_img[i],
                hidden_space_dim=self.num_tokens[i],
                norm_layer=nn.Identity,
                channel_first=True,
                space_dim=self.space_dim[i],
            )
            for i in range(len(dim_img))
        ])

        self.out_convs = nn.ModuleList([
            nn.Conv2d(in_channels=dim_img[i] * 2, out_channels=dim_img[i], kernel_size=1)
            for i in range(len(dim_img))
        ])

        from models.decoder.segformer_decoder import SegFormerHead
        self.decoder = SegFormerHead(in_channels=dim_img, num_classes=num_classes)

    @staticmethod
    def _run_stage_blocks(blocks, x):
        # x can be tokens [B, N, C] or stage-0 NCHW [B, C, H, W]
        if x.dim() == 4:
            b, c, h, w = x.shape
            tokens = x.flatten(2).transpose(1, 2).contiguous()
        else:
            b, n, c = x.shape
            h = w = None
            tokens = x

        if h is None or w is None:
            n = tokens.shape[1]
            raise RuntimeError("Stage-1 input should be NCHW; later-stage inputs are handled after downsample.")

        for blk in blocks:
            tokens = blk(tokens, h, w)
        feat = tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()
        return feat

    @staticmethod
    def _blocks_from_tokens(blocks, x_tokens, h, w):
        x = x_tokens
        for blk in blocks:
            x = blk(x, h, w)
        b, _, c = x.shape
        feat = x.transpose(1, 2).reshape(b, c, h, w).contiguous()
        return feat, x

    def forward(self, x_ev, x_img):
        return self.forward_fuse_v2(x_ev, x_img)

    def forward_fuse_v2(self, x_ev, x_img):
        _, _, H0, W0 = x_ev.shape
        outs = []

        # stage-0 stem output is tokenized inside Stem, so we immediately restore NCHW.
        x_ev_tokens, h0_ev, w0_ev = self.encoder_ev.patch_embed(x_ev)
        x_img_tokens, h0_img, w0_img = self.encoder_img.patch_embed(x_img)
        x_ev = x_ev_tokens.transpose(1, 2).reshape(
            x_ev_tokens.shape[0], self.encoder_ev.dims[0], h0_ev, w0_ev
        ).contiguous()
        x_img = x_img_tokens.transpose(1, 2).reshape(
            x_img_tokens.shape[0], self.encoder_img.dims[0], h0_img, w0_img
        ).contiguous()

        for i, (layer_ev, layer_img) in enumerate(zip(self.encoder_ev.layers, self.encoder_img.layers)):
            if i == 0:
                h, w = x_ev.shape[-2], x_ev.shape[-1]
                o_ev_tokens = x_ev.flatten(2).transpose(1, 2).contiguous()
                o_img_tokens = x_img.flatten(2).transpose(1, 2).contiguous()
            else:
                h, w = self.space_dim[i]
                o_ev_tokens = x_ev
                o_img_tokens = x_img

            out_ev, o_ev_tokens = self._blocks_from_tokens(layer_ev.blocks, o_ev_tokens, h, w)
            out_img, o_img_tokens = self._blocks_from_tokens(layer_img.blocks, o_img_tokens, h, w)

            norm_layer_ev = getattr(self.encoder_ev, f'outnorm{i}')
            norm_layer_img = getattr(self.encoder_img, f'outnorm{i}')
            out_ev = norm_layer_ev(out_ev)
            out_img = norm_layer_img(out_img)

            img, ev = self.ccbs[i](out_img, out_ev)
            concat_out = torch.cat((img, ev), dim=1)
            outs.append(self.out_convs[i](concat_out))

            if i < len(self.encoder_ev.layers) - 1:
                x_ev = layer_ev.downsample(
                    o_ev_tokens.transpose(1, 2).reshape(
                        o_ev_tokens.shape[0], self.encoder_ev.dims[i], h, w
                    ).contiguous() + ev
                )
                x_img = layer_img.downsample(
                    o_img_tokens.transpose(1, 2).reshape(
                        o_img_tokens.shape[0], self.encoder_img.dims[i], h, w
                    ).contiguous() + img
                )

        x = self.decoder(outs)
        x = F.interpolate(x, size=[H0, W0], mode='bilinear', align_corners=False)
        return x
