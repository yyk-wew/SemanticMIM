from mmpretrain import get_model
import beit
import pdb

import os
import sys
import argparse
import random
import colorsys
import requests
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--num_cls_tokens', default=8, type=int)
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--image_path", default='/data/yike/imagenet/train/n04548362/n04548362_9084.JPEG', type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='./vis/ours/', help='Path where to save visualizations.')
    parser.add_argument("--pretrained_weights", default='/data/yike/checkpoint/ours_8_cls_300_epoch', type=str)
    parser.add_argument("--get_cls_attn", action='store_true')
    args = parser.parse_args()

    print(args.get_cls_attn)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # ============ building network ... ============
    if 'ours' in args.pretrained_weights:
        model = get_model(model='configs/beit/beit_bottleneck_pretrain.py', pretrained=args.pretrained_weights, backbone=dict(out_indices=[0], num_cls_tokens=args.num_cls_tokens))
    elif 'baseline' in args.pretrained_weights:
        model = get_model(model='configs/beit/beit_baseline_pretrain.py', pretrained=args.pretrained_weights, backbone=dict(out_indices=[0], num_cls_tokens=args.num_cls_tokens))

    print(model.backbone.out_indices)

    model.cuda()
    model.eval()
    # print(f"Model {args.model} built.")

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
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
        attentions = model.backbone.get_last_selfattention(img.to(device))[0]

    # import pdb
    # pdb.set_trace()

    nh = attentions.shape[1] # number of head
    print("shape", attentions.shape)

    # # calculate cls token weight
    # inter_cls_token_weight = attentions[0, :, args.num_cls_tokens:args.num_cls_tokens+5, :args.num_cls_tokens]
    # inter_cls_token_weight = inter_cls_token_weight.mean(dim=0)
    # print(inter_cls_token_weight)
    # raise RuntimeError()

    mean_cls = None
    if args.get_cls_attn:
        attentions = attentions[0, :, :args.num_cls_tokens, args.num_cls_tokens:]
        seq_num = args.num_cls_tokens
    else:
        # # obtain avg cls feat map
        # cls_attn = attentions[0, :, :args.num_cls_tokens, args.num_cls_tokens:]
        # mean_cls = cls_attn.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True).reshape(1, 1, w_featmap, h_featmap)
        # mean_cls = nn.functional.interpolate(mean_cls, scale_factor=args.patch_size, mode="nearest").cpu().numpy()

        indices = [0, 200, 400, 600, 899]
        attentions = attentions[0, :, :, args.num_cls_tokens:]
        attentions = torch.index_select(attentions, dim=1, index=torch.tensor(indices).cuda())
        seq_num = len(indices)
    attentions = attentions.reshape(nh, seq_num, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions, scale_factor=args.patch_size, mode="nearest").cpu().numpy()

    # save attentions heatmaps
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.basename(args.image_path)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, base_name))
    base_name = os.path.splitext(base_name)[0]
    postfix = "-attn-avg.png" if args.get_cls_attn else '-patch-avg.png'

    if mean_cls is not None:
        mean_cls = mean_cls[0][0]
        fname = os.path.join(args.output_dir, base_name + '-mean-cls' + postfix)
        plt.imsave(fname=fname, arr=mean_cls, format='png')


    # Plot each head and an avg attention map
    for i in range(seq_num):
        temp_attn = attentions[:, i]
        # for j in range(nh):
        #     fname = os.path.join(args.output_dir, base_name + '-' + str(i) + "-attn-head" + str(j) + ".png")
        #     plt.imsave(fname=fname, arr=temp_attn[j], format='png')
        #     print(f"{fname} saved.")
        fname = os.path.join(args.output_dir, base_name + '-' + str(i) + postfix)
        temp_attn = np.mean(temp_attn, axis=0)
        if mean_cls is not None:
            temp_attn = temp_attn - mean_cls
        plt.imsave(fname=fname, arr=temp_attn, format='png')


    mean_attn = np.mean(attentions, axis=0) # shape [query, w, h]
    num, w, h = mean_attn.shape
    attn_as_row = np.concatenate(np.split(mean_attn, num, axis=0), axis=2)[0]
    fname = os.path.join(args.output_dir, base_name + postfix)
    plt.imsave(fname=fname, arr=attn_as_row, format='png')
    