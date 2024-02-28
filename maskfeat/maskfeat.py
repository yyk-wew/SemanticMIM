from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import numpy as np

from mmpretrain.models import BaseSelfSupervisor
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmengine.model import ModuleList

from .maskfeat_backbone import TransformerEncoderLayerOurs, VisionTransformerOurs


@MODELS.register_module()
class MaskFeatViTOurs(VisionTransformerOurs):
    """Vision Transformer for MaskFeat pre-training.

    A PyTorch implement of: `Masked Feature Prediction for Self-Supervised
    Visual Pre-Training <https://arxiv.org/abs/2112.09133>`_.

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
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

            It only works without input mask. Defaults to ``"avg_featmap"``.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch: Union[str, dict] = 'b',
                 img_size: int = 224,
                 patch_size: int = 16,
                 out_indices: Union[Sequence, int] = -1,
                 drop_rate: float = 0,
                 drop_path_rate: float = 0,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 final_norm: bool = True,
                 out_type: str = 'raw',
                 interpolate_mode: str = 'bicubic',
                 use_bottleneck: bool = False,
                 num_cls_tokens: int = 1,
                 patch_cfg: dict = dict(),
                 layer_cfgs: dict = dict(),
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            out_type=out_type,
            num_cls_tokens=num_cls_tokens,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)

        self.mask_token = nn.parameter.Parameter(
            torch.zeros(1, 1, self.embed_dims), requires_grad=True)
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        self.use_bottleneck = use_bottleneck
        if use_bottleneck:
            self.extra_layers = ModuleList()
            if isinstance(layer_cfgs, dict):
                layer_cfgs = [layer_cfgs] * self.num_layers
            dpr = np.linspace(0, drop_path_rate, self.num_layers)
            for i in range(self.num_layers):
                _layer_cfg = dict(
                    embed_dims=self.embed_dims,
                    num_heads=self.arch_settings['num_heads'],
                    feedforward_channels=self.
                    arch_settings['feedforward_channels'],
                    layer_scale_init_value=0.,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    qkv_bias=True,
                    norm_cfg=norm_cfg)
                _layer_cfg.update(layer_cfgs[i])
                self.extra_layers.append(TransformerEncoderLayerOurs(**_layer_cfg))

    def init_weights(self) -> None:
        """Initialize position embedding, mask token and cls token."""
        super().init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):

            nn.init.trunc_normal_(self.cls_token, std=.02)
            nn.init.trunc_normal_(self.mask_token, std=.02)
            nn.init.trunc_normal_(self.pos_embed, std=.02)

            self.apply(self._init_weights)

    def _init_weights(self, m: torch.nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor,
                img_index: Optional[torch.Tensor],
                mask_index: Optional[torch.Tensor]) -> torch.Tensor:
        """Generate features for masked images.

        The function supports two kind of forward behaviors. If the ``mask`` is
        not ``None``, the forward function will be executed as masked image
        modeling pre-training; if the ``mask`` is ``None``, the forward
        function will call ``super().forward()``, which extract features from
        images without mask.

        Args:
            x (torch.Tensor): Input images.
            mask (torch.Tensor, optional): Input masks.

        Returns:
            torch.Tensor: Features with cls_tokens.
        """
        if img_index is None:
            return super().forward(x)

        else:
            # B = x.shape[0]
            x = self.patch_embed(x)[0]
            B, L, D = x.shape

            # masking: length -> length * mask_ratio
            # B, L, _ = x.shape
            # mask_tokens = self.mask_token.expand(B, L, -1)
            # mask = mask.unsqueeze(-1)
            # x = x * (1 - mask.int()) + mask_tokens * mask

            # append cls token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.drop_after_pos(x)

            cls_tokens = x[:, :self.num_extra_tokens]
            other_tokens = x[:, self.num_extra_tokens:]
            image_tokens = other_tokens[img_index].reshape(B, -1, D)
            num_img_tokens = image_tokens.shape[1]

            mask_tokens = self.mask_token.expand(B, L, -1)
            mask_tokens = mask_tokens + self.pos_embed[:, self.num_extra_tokens:]
            mask_tokens = mask_tokens[mask_index].reshape(B, -1, D)
            num_masked_tokens = mask_tokens.shape[1]

            x = torch.cat((cls_tokens, image_tokens, mask_tokens), dim=1)
            for i in range(len(self.layers)):
                if self.use_bottleneck:
                    x_cls_img = torch.cat((cls_tokens, image_tokens), dim=1)
                    x_cls_img = self.layers[i](q=x_cls_img, kv=x_cls_img)
                    cls_tokens, image_tokens = torch.split(x_cls_img, [self.num_extra_tokens, num_img_tokens], dim=1)
                    
                    x_cls_mask = torch.cat((cls_tokens, mask_tokens), dim=1)
                    mask_tokens = self.extra_layers[i](q=mask_tokens, kv=x_cls_mask)
                else:
                    x = self.layers[i](q=x, kv=x)

            if self.use_bottleneck:
                x = torch.cat((cls_tokens, image_tokens, mask_tokens), dim=1)

            if self.final_norm:
                x = self.norm1(x)

            return x[:, -num_masked_tokens:]


@MODELS.register_module()
class MaskFeatOurs(BaseSelfSupervisor):
    """MaskFeat.

    Implementation of `Masked Feature Prediction for Self-Supervised Visual
    Pre-Training <https://arxiv.org/abs/2112.09133>`_.
    """

    def extract_feat(self, inputs: torch.Tensor):
        return self.backbone(inputs, mask=None)

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (torch.Tensor): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        img_index = torch.stack([data_sample.img_index for data_sample in data_samples])
        img_index = img_index.flatten(1).bool()
        mask_index = torch.stack([data_sample.mask_index for data_sample in data_samples])
        mask_index = mask_index.flatten(1).bool()

        latent = self.backbone(inputs, img_index, mask_index)
        B, L, C = latent.shape
        pred = self.neck((latent.reshape(B * L, C), ))
        pred = pred[0].reshape(B, L, -1)
        hog = self.target_generator(inputs)
        hog = hog[mask_index].reshape(B, L, -1)

        # remove cls_token before compute loss
        loss = self.head.loss(pred, hog)
        losses = dict(loss=loss)
        return losses