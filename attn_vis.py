from mmpretrain import get_model

import os
import sys
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import maskfeat
import beit


def draw_attention(attentions, args, layer):
    # attentions [nh, query_seq, kv_seq]
    nh = attentions.shape[0] # number of head

    mean_cls = None
    if args.get_cls_attn:
        attentions = attentions[:, :args.num_cls_tokens, args.num_cls_tokens:]
        seq_num = args.num_cls_tokens
        indices = list(range(seq_num))
    else:
        if args.reduce_mean:
            # obtain avg cls feat map
            cls_attn = attentions[:, :args.num_cls_tokens, args.num_cls_tokens:]
            mean_cls = cls_attn.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True).reshape(1, 1, w_featmap, h_featmap)
            mean_cls = nn.functional.interpolate(mean_cls, scale_factor=args.patch_size, mode="nearest").cpu().numpy()

        indices = args.token_indices
        attentions = attentions[:, args.num_cls_tokens:, args.num_cls_tokens:]
        attentions = torch.index_select(attentions, dim=1, index=torch.tensor(indices).cuda())
        seq_num = len(indices)
    attentions = attentions.reshape(nh, seq_num, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions, scale_factor=args.patch_size, mode="nearest").cpu().numpy()

    # save attentions heatmaps
    
    post_dir = "cls" if args.get_cls_attn else 'patch'

    os.makedirs(os.path.join(args.output_dir, base_name, post_dir), exist_ok=True)

    if mean_cls is not None:
        mean_cls = mean_cls[0][0]
        fname = os.path.join(args.output_dir, base_name, post_dir, 'layer-' + str(layer) + '-mean-cls' + '.png')
        plt.imsave(fname=fname, arr=mean_cls, format='png')

    for i, index in enumerate(indices):
        temp_attn = attentions[:, i]
        postfix = 'reduce.png' if mean_cls is not None else '.png'
        fname = os.path.join(args.output_dir, base_name, post_dir, 'layer-' + str(layer) + '-token-' + str(index) + postfix)
        temp_attn = np.mean(temp_attn, axis=0)
        if mean_cls is not None:
            temp_attn = temp_attn - mean_cls
        plt.imsave(fname=fname, arr=temp_attn, format='png')

    mean_attn = np.mean(attentions, axis=0) # shape [query, w, h]
    num, w, h = mean_attn.shape
    attn_as_row = np.concatenate(np.split(mean_attn, num, axis=0), axis=2)[0]
    fname = os.path.join(args.output_dir, base_name, post_dir, 'layer-' + str(layer) + '-all.png')
    plt.imsave(fname=fname, arr=attn_as_row, format='png')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--num_cls_tokens', default=8, type=int)
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--image_path", default='/data/yike/imagenet/train/n04548362/n04548362_9084.JPEG', type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='./vis/ours/', help='Path where to save visualizations.')
    parser.add_argument("--pretrained_weights", default='/data/yike/checkpoint/ours_8_cls_300_epoch', type=str)
    parser.add_argument("--get_cls_attn", action='store_true', help='Whether to visualize attention of [CLS] token or patch tokens.')
    parser.add_argument("--reduce_mean", action='store_true', help='Whether to reduce averaged [CLS] token attention.')
    parser.add_argument("--layer_indices", default=(-1, ), type=int, nargs="+", help="layer index list. -1 represents all layers.")
    parser.add_argument("--token_indices", default=(0, ), type=int, nargs="+", help="token index used as queries.")
    args = parser.parse_args()

    print("Get cls token attn: ", args.get_cls_attn)
    print("Use Reduce: ", args.reduce_mean)
    if args.layer_indices[0] == -1:
        args.layer_indices = tuple(range(0, 12))
    print("Layer indices: ", args.layer_indices)
    print("Token indices: ", args.token_indices)
    

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # ============ building network ... ============
    if 'ours' in args.pretrained_weights:
        model = get_model(model='configs/beit/beit_bottleneck_pretrain.py', pretrained=args.pretrained_weights, backbone=dict(out_indices=args.layer_indices, num_cls_tokens=args.num_cls_tokens))
    elif 'baseline' in args.pretrained_weights:
        model = get_model(model='configs/beit/beit_baseline_pretrain.py', pretrained=args.pretrained_weights, backbone=dict(out_indices=args.layer_indices, num_cls_tokens=args.num_cls_tokens))
    elif 'moco' in args.pretrained_weights:
        model = get_model(model='./moco_config.py', backbone=dict(out_indices=args.layer_indices, num_cls_tokens=args.num_cls_tokens))

    model.cuda()
    model.eval()

    # open image
    if os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    model.eval()
    with torch.no_grad():
        attentions = model.backbone.get_last_selfattention(img.to(device))

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.basename(args.image_path)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, base_name))
    base_name = os.path.splitext(base_name)[0]

    for i in range(len(attentions)):
        draw_attention(attentions[i][0], args, i)