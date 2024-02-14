import numpy as np
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Dict

from .beit_backbone import BEiTViTOurs, BEiTTransformerEncoderLayer
from mmpretrain.models.utils import resize_pos_embed
from mmengine.model.weight_init import trunc_normal_
from mmpretrain.registry import MODELS, TRANSFORMS
from mmengine.model import BaseModule, ModuleList
from mmpretrain.models.selfsup import BaseSelfSupervisor
from mmpretrain.structures import DataSample
from mmcv.transforms import BaseTransform

@MODELS.register_module()
class BEiTPretrainViTOurs(BEiTViTOurs):
    """Vision Transformer for BEiT pre-training.

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base' and 'large'. If use dict, it should
            have below keys:

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
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
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
        num_cls_tokens (bool): Number of [CLS] tokens of ViT. Defaults to 1.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        layer_scale_init_value (float): The initialization value for
            the learnable scaling of attention and FFN. Defaults to 0.1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        use_bottleneck (bool): Whether to use bottleneck Transformer for masked
            image modeling pretrain. Defaults to False.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    def __init__(self,
                 arch: str = 'base',
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 out_indices: int = -1,
                 drop_rate: float = 0,
                 drop_path_rate: float = 0,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 final_norm: bool = True,
                 out_type: str = 'raw',
                 num_cls_tokens: int = 1,
                 frozen_stages: int = -1,
                 layer_scale_init_value: int = 0.1,
                 interpolate_mode: str = 'bicubic',
                 use_bottleneck: bool = False,
                 patch_cfg: dict = dict(padding=0),
                 layer_cfgs: dict = dict(),
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            out_type=out_type,
            num_cls_tokens=num_cls_tokens,
            frozen_stages=frozen_stages,
            layer_scale_init_value=layer_scale_init_value,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        self.use_bottleneck = use_bottleneck

        if use_bottleneck:
            self.extra_layers = ModuleList()
            dpr = np.linspace(0, drop_path_rate, self.num_layers)
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
                    bias='qv_bias',
                    norm_cfg=norm_cfg)
                _layer_cfg.update(layer_cfgs[i])
                self.extra_layers.append(BEiTTransformerEncoderLayer(**_layer_cfg))



    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding and cls token."""
        super().init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)
        self.rescale_init_weight()

    def rescale_init_weight(self) -> None:
        """Rescale the initialized weights."""

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.layers):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.ffn.layers[1].weight.data, layer_id + 1)

        if self.use_bottleneck:
            for layer_id, layer in enumerate(self.extra_layers):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.ffn.layers[1].weight.data, layer_id + 1)

    def forward(self, x: torch.Tensor,
                img_index: Optional[torch.Tensor],
                mask_index: Optional[torch.Tensor]) -> Tuple[torch.Tensor]:
        """The BEiT style forward function.

        The function supports two kind of forward behaviors. If the ``mask`` is
        not ``None``, the forward function will be executed as masked image
        modeling pre-training; if the ``mask`` is ``None``, the forward
        function will call ``super().forward()``, which extract features from
        images without mask.

        Args:
            x (torch.Tensor): Input images, which is of shape (B x C x H x W).
            mask (torch.Tensor, optional): Mask for input, which is of shape
                (B x patch_resolution[0] x patch_resolution[1]).

        Returns:
            Tuple[torch.Tensor]: Hidden features.
        """
        if img_index is None:
            return super().forward(x)

        else:
            x, patch_resolution = self.patch_embed(x)

            # # replace the masked visual tokens by mask_token
            # B, L, _ = x.shape
            # mask_token = self.mask_token.expand(B, L, -1)
            # w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            # x = x * (1. - w) + mask_token * w

            # stole cls_tokens impl from Phil Wang, thanks
            B, L, D = x.shape
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            if self.pos_embed is not None:
                resized_pos_embed = resize_pos_embed(
                    self.pos_embed,
                    self.patch_resolution,
                    patch_resolution,
                    mode=self.interpolate_mode,
                    num_extra_tokens=self.num_extra_tokens)
                x = x + resized_pos_embed
            x = self.drop_after_pos(x)

            # build mask tokens
            cls_tokens = x[:, :self.num_extra_tokens]
            other_tokens = x[:, self.num_extra_tokens:]
            image_tokens = other_tokens[img_index.flatten(1).to(torch.bool)].reshape(B, -1, D)
            num_img_tokens = image_tokens.shape[1]

            mask_tokens = self.mask_token.expand(B, L, -1)
            if self.pos_embed is not None:
                mask_tokens = mask_tokens + resized_pos_embed[:, self.num_extra_tokens:]
            mask_tokens = mask_tokens[mask_index.flatten(1).to(torch.bool)].reshape(B, -1, D)
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
            
            return (x[:, -num_masked_tokens:], )



@MODELS.register_module()
class BEiTV1HeadOurs(BaseModule):
    """Head for BEiT v1 Pre-training.

    Compute the logits and the cross entropy loss.

    Args:
        embed_dims (int): The dimension of embedding.
        num_embed (int): The number of classification types.
        loss (dict): The config of loss.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_embed: int,
        loss: dict,
        init_cfg: Optional[Union[dict, List[dict]]] = dict(
            type='TruncNormal', layer='Linear', std=0.02, bias=0)
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.cls_head = nn.Linear(embed_dims, num_embed)
        self.loss_module = MODELS.build(loss)

    def loss(self, feats: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        """Generate loss.

        Args:
            feats (torch.Tensor): Features from backbone.
            target (torch.Tensor): Target generated by target_generator.
            mask (torch.Tensor): Generated mask for pretraing.
        """
        mask = mask.flatten(1).to(torch.bool)
        target = torch.argmax(target, dim=1).flatten(1)
        target = target[mask]

        logits = self.cls_head(feats).flatten(0,1)

        loss = self.loss_module(logits, target)
        return loss


@MODELS.register_module()
class BEiTOurs(BaseSelfSupervisor):
    """BEiT v1/v2.

    Implementation of `BEiT: BERT Pre-Training of Image Transformers
    <https://arxiv.org/abs/2106.08254>`_ and `BEiT v2: Masked Image Modeling
    with Vector-Quantized Visual Tokenizers
    <https://arxiv.org/abs/2208.06366>`_.
    """

    def extract_feat(self, inputs: torch.Tensor):
        return self.backbone(inputs, mask=None)

    def loss(self, inputs: List[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        img_index = torch.stack([data_sample.img_index for data_sample in data_samples])
        mask_index = torch.stack([data_sample.mask_index for data_sample in data_samples])

        img_latent = self.backbone(inputs[0], img_index, mask_index)

        # inputs[1] is the target image
        with torch.no_grad():
            target = self.target_generator(inputs[1])
            target = target.detach()

        # BEiT v1
        loss = self.head.loss(img_latent[0], target, mask_index)

        if isinstance(loss, torch.Tensor):
            losses = dict(loss=loss)
            return losses
        elif isinstance(loss, Tuple):
            # the loss_1 and loss_2 are general reconstruction loss (patch
            # feature vectors from last layer of backbone) and early state
            # reconstruction loss (patch feature vectors from intermediate
            # layer of backbone)
            loss_1, loss_2 = loss[0], loss[1]
            losses = dict()
            # the key with prefix 'loss', like loss_1 and loss_2, will be used
            # as the final criterion
            losses['loss_1'] = loss_1
            losses['loss_2'] = loss_2
            return losses

@TRANSFORMS.register_module()
class SimMIMMaskGeneratorOurs(BaseTransform):
    """Generate random block mask for each Image.

    **Added Keys**:

    - mask

    This module is used in SimMIM to generate masks.

    Args:
        input_size (int): Size of input image. Defaults to 192.
        mask_patch_size (int): Size of each block mask. Defaults to 32.
        model_patch_size (int): Patch size of each token. Defaults to 4.
        mask_ratio (float): The mask ratio of image. Defaults to 0.6.
    """

    def __init__(self,
                 input_size: int = 192,
                 mask_patch_size: int = 32,
                 model_patch_size: int = 4,
                 mask_ratio: float = 0.6,
                 use_separate_mask: bool = False, 
                 separate_mask_ratio: float = 0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.use_separate_mask = use_separate_mask
        self.separate_mask_ratio = separate_mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        self.separate_mask_count = int(np.ceil(self.token_count * self.separate_mask_ratio))

    def transform(self, results: dict) -> dict:
        """Method to generate random block mask for each Image in SimMIM.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with added key ``mask``.
        """
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        mask = torch.from_numpy(mask)

        results.update({'img_index': 1. - mask})

        if self.use_separate_mask:
            sep_mask_idx = np.random.permutation(self.token_count)[:self.separate_mask_count]
            sep_mask = np.zeros(self.token_count, dtype=int)
            sep_mask[sep_mask_idx] = 1

            sep_mask = sep_mask.reshape((self.rand_size, self.rand_size))
            sep_mask = sep_mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
            sep_mask = torch.from_numpy(sep_mask)

            results.update({'mask_index': sep_mask})
        else:
            results.update({'mask_index': mask})

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'mask_patch_size={self.mask_patch_size}, '
        repr_str += f'model_patch_size={self.model_patch_size}, '
        repr_str += f'mask_ratio={self.mask_ratio})'
        return repr_str