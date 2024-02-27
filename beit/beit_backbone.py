from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.registry import MODELS
from mmpretrain.models.utils import build_norm_layer, resize_pos_embed, to_2tuple
from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.models.backbones.vision_transformer import TransformerEncoderLayer

from mmseg.registry import MODELS as MODELS_SEG


class BEiTAttention(BaseModule):
    """Original multi-head attention with q and kv splitted 
     for support of self-attention and cross-attention.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        bias (str): The option to add leanable bias for q, k, v. If bias is
            True, it will add leanable bias. If bias is 'qv_bias', it will only
            add leanable bias for q, v. If bias is False, it will not add bias
            for q, k, v. Default to 'qv_bias'.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 bias='qv_bias',
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.bias = bias
        self.scale = qk_scale or head_embed_dims**-0.5

        if bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(embed_dims))
        else:
            self.q_bias = None
            self.v_bias = None
            
        self.q_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.kv_proj = nn.Linear(embed_dims, embed_dims*2, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)


    def forward(self, q, kv, return_attn=False):
        """
        Args:
            x (tensor): input features with shape of (num_windows*B, N, C).
            rel_pos_bias (tensor): input relative position bias with shape of
                (num_heads, N, N).
        """

        B, N_q, _ = q.shape
        _, N_kv, _ = kv.shape
        kv_bias = None
        if self.q_bias is not None:
            kv_bias = torch.cat((torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            
        q = F.linear(input=q, weight=self.q_proj.weight, bias=self.q_bias).reshape(B, N_q, self.num_heads, -1).permute(0, 2, 1, 3)
        kv = F.linear(input=kv, weight=self.kv_proj.weight, bias=kv_bias).reshape(B, N_kv, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)

        if return_attn:
            return attn

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BEiTTransformerEncoderLayer(TransformerEncoderLayer):
    """Implements one encoder layer in BEiT.

    Comparing with conventional ``TransformerEncoderLayer``, this module
    adds weights to the shortcut connection. In addition, ``BEiTAttention``
    is used to replace the original ``MultiheadAttention`` in
    ``TransformerEncoderLayer``.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        layer_scale_init_value (float): The initialization value for
            the learnable scaling of attention and FFN. 1 means no scaling.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        window_size (tuple[int]): The height and width of the window.
            Defaults to None.
        use_rel_pos_bias (bool): Whether to use unique relative position bias,
            if False, use shared relative position bias defined in backbone.
        attn_drop_rate (float): The drop out rate for attention layer.
            Defaults to 0.0.
        drop_path_rate (float): Stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        bias (bool | str): The option to add leanable bias for q, k, v. If bias
            is True, it will add leanable bias. If bias is 'qv_bias', it will
            only add leanable bias for q, v. If bias is False, it will not add
            bias for q, k, v. Default to 'qv_bias'.
        act_cfg (dict): The activation config for FFNs.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='LN').
        attn_cfg (dict): The configuration for the attention layer.
            Defaults to an empty dict.
        ffn_cfg (dict): The configuration for the ffn layer.
            Defaults to ``dict(add_identity=False)``.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 layer_scale_init_value: float,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 bias: Union[str, bool] = 'qv_bias',
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 attn_cfg: dict = dict(),
                 ffn_cfg: dict = dict(add_identity=False),
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=0.,
            drop_rate=0.,
            num_fcs=num_fcs,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)

        attn_cfg = {
            'qk_scale': None,
            'embed_dims': embed_dims,
            'num_heads': num_heads,
            'attn_drop': attn_drop_rate,
            'proj_drop': drop_rate,
            'bias': bias,
            **attn_cfg,
        }
        self.attn = BEiTAttention(**attn_cfg)

        ffn_cfg = {
            'embed_dims': embed_dims,
            'feedforward_channels': feedforward_channels,
            'num_fcs': num_fcs,
            'ffn_drop': drop_rate,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path_rate),
            'act_cfg': act_cfg,
            **ffn_cfg,
        }
        self.ffn = FFN(**ffn_cfg)

        # NOTE: drop path for stochastic depth, we shall see if
        # this is better than dropout here
        dropout_layer = dict(type='DropPath', drop_prob=drop_path_rate)
        self.drop_path = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

        if layer_scale_init_value > 0:
            self.gamma_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((embed_dims)),
                requires_grad=True)
            self.gamma_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((embed_dims)),
                requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        if self.gamma_1 is None:
            q = q + self.drop_path(
                self.attn(q=self.ln1(q), kv=self.ln1(kv)))
            q = q + self.drop_path(self.ffn(self.ln2(q)))
        else:
            q = q + self.drop_path(self.gamma_1 * self.attn(
                q=self.ln1(q), kv=self.ln1(kv)))
            q = q + self.drop_path(self.gamma_2 * self.ffn(self.ln2(q)))
        return q
    
    def get_attn(self, q, kv):
        return self.attn(q=self.ln1(q), kv=self.ln1(kv), return_attn=True)


@MODELS_SEG.register_module()
@MODELS.register_module()
class BEiTViTOurs(BaseBackbone):
    """Backbone for BEiT.

    A PyTorch implement of : `BEiT: BERT Pre-Training of Image Transformers
    <https://arxiv.org/abs/2106.08254>`_
    A PyTorch implement of : `BEiT v2: Masked Image Modeling with
    Vector-Quantized Visual Tokenizers <https://arxiv.org/abs/2208.06366>`_

    Args:
        arch (str | dict): BEiT architecture. If use string, choose from
            'base', 'large'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        bias (bool | str): The option to add leanable bias for q, k, v. If bias
            is True, it will add leanable bias. If bias is 'qv_bias', it will
            only add leanable bias for q, v. If bias is False, it will not add
            bias for q, k, v. Default to 'qv_bias'.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        out_type (str): The type of output features. Please choose from

            - ``"cls_token"``: The class token tensor with shape (B, C).
            - ``"featmap"``: The feature map tensor from the patch tokens
              with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).
            - ``"raw"``: The raw feature tensor includes patch tokens and
              class tokens with shape (B, L, C).

            Defaults to ``"avg_featmap"``.
        num_cls_tokens (bool): Number of [CLS] tokens of ViT. Defaults to 1.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        use_abs_pos_emb (bool): Use position embedding like vanilla ViT.
            Defaults to False.
        use_rel_pos_bias (bool): Use relative position embedding in each
            transformer encoder layer. Defaults to True.
        use_shared_rel_pos_bias (bool): Use shared relative position embedding,
            all transformer encoder layers share the same relative position
            embedding. Defaults to False.
        layer_scale_init_value (float): The initialization value for
            the learnable scaling of attention and FFN. Defaults to 0.1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
        **dict.fromkeys(
            ['eva-g', 'eva-giant'],
            {
                # The implementation in EVA
                # <https://arxiv.org/abs/2211.07636>
                'embed_dims': 1408,
                'num_layers': 40,
                'num_heads': 16,
                'feedforward_channels': 6144
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': 192 * 4
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
    }
    num_extra_tokens = 1  # class token
    OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap'}

    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0,
                 drop_path_rate=0,
                 bias='qv_bias',
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=False,
                 out_type='avg_featmap',
                 num_cls_tokens=1,
                 frozen_stages=-1,
                 interpolate_mode='bicubic',
                 layer_scale_init_value=0.1,
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super(BEiTViTOurs, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = to_2tuple(img_size)

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        # Set cls token
        self.num_extra_tokens = num_cls_tokens
        if num_cls_tokens > 0:
            self.cls_token = nn.Parameter(torch.zeros(1, self.num_extra_tokens, self.embed_dims))
        elif out_type != 'cls_token' and self.num_extra_tokens == 0:
            self.cls_token = None
        else:
            raise ValueError(
                'num_cls_tokens must be positive int when `out_type="cls_token"`.')

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_extra_tokens,
                        self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                layer_scale_init_value=layer_scale_init_value,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                bias=bias,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(BEiTTransformerEncoderLayer(**_layer_cfg))

        self.frozen_stages = frozen_stages
        self.final_norm = final_norm
        if final_norm:
            self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        if out_type in ['avg_featmap', 'cls_token']:
            self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        super(BEiTViTOurs, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.pos_embed.shape != ckpt_pos_embed_shape:
            logger.info(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.')

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze cls_token
        if self.num_extra_tokens > 0:
            self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers):
            if self.final_norm:
                self.ln1.eval()
                for param in self.ln1.parameters():
                    param.requires_grad = False

            if self.out_type in ['avg_featmap', 'cls_token']:
                self.ln2.eval()
                for param in self.ln2.parameters():
                    param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.pos_embed is not None:
            x = x + resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(q=x, kv=x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)

    def _format_output(self, x, hw):
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            return self.ln2(x[:, :self.num_extra_tokens].mean(dim=1))

        patch_token = x[:, self.num_extra_tokens:]
        if self.out_type == 'featmap':
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2).contiguous()
        if self.out_type == 'avg_featmap':
            return self.ln2(patch_token.mean(dim=1))

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.

        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.num_layers + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix):]

        if param_name in ('cls_token', 'pos_embed'):
            layer_depth = 0
        elif param_name.startswith('patch_embed'):
            layer_depth = 0
        elif param_name.startswith('layers'):
            layer_id = int(param_name.split('.')[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers

    def get_last_selfattention(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.pos_embed is not None:
            x = x + resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=self.num_extra_tokens)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(q=x, kv=x)

            if i in self.out_indices:
                outs.append(self.layers[i].get_attn(q=x, kv=x))

        return tuple(outs)