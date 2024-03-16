import os
from random import shuffle
import sys
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from torchvision.transforms.functional import InterpolationMode

import voc_utils.utils as utils

from voc_utils.segment_utils import VOCDataset, PredsmIoU, StreamSegMetrics
from voc_utils.segment_utils import Compose, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Normalize
from tqdm import tqdm

from mmpretrain import get_model
import beit
import maskfeat


def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    if 'maskfeat' in args.pretrained_weights:
        if 'ours' in args.pretrained_weights:
            print("load maskfeat ours")
            model = get_model(model='configs/maskfeat/maskfeat_bottleneck_pretrain.py', pretrained=args.pretrained_weights, backbone=dict(img_size=args.input_size, out_indices=args.layer_indices, num_cls_tokens=args.num_cls_tokens, out_type='featmap'))
        else:
            print("load maskfeat baseline")
            model = get_model(model='configs/maskfeat/maskfeat_baseline_pretrain.py', pretrained=args.pretrained_weights, backbone=dict(img_size=args.input_size, out_indices=args.layer_indices, num_cls_tokens=args.num_cls_tokens, out_type='featmap'))
    else:
        if 'ours' in args.pretrained_weights:
            print("Load beit ours")
            model = get_model(model='configs/beit/beit_bottleneck_pretrain.py', pretrained=args.pretrained_weights, backbone=dict(img_size=args.input_size, out_indices=args.layer_indices, num_cls_tokens=args.num_cls_tokens, out_type='featmap'))
        else:
            print("load beit baseline")
            model = get_model(model='configs/beit/beit_baseline_pretrain.py', pretrained=args.pretrained_weights, backbone=dict(img_size=args.input_size, out_indices=args.layer_indices, num_cls_tokens=args.num_cls_tokens, out_type='featmap'))
        

    model.cuda()
    model.eval()
    
    if args.head_type == 'linear':
        seg_head = nn.Conv2d(model.backbone.embed_dims, args.num_classes, 1)
        nn.init.normal_(seg_head.weight, mean=0, std=0.01)
        nn.init.zeros_(seg_head.bias)
    elif args.head_type == 'fcn':
        pass
    
    seg_head = seg_head.cuda()
    seg_head = nn.parallel.DistributedDataParallel(seg_head, device_ids=[args.gpu])

    # ============ preparing data ... ============
    val_image_transform = pth_transforms.Compose([
        pth_transforms.Resize((args.input_size, args.input_size)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_target_transform = pth_transforms.Compose([
        pth_transforms.Resize((args.input_size, args.input_size), interpolation=InterpolationMode.NEAREST),
        pth_transforms.ToTensor(),
    ])
    dataset_val = VOCDataset(root=args.data_path, image_set='val', transform=val_image_transform, target_transform=val_target_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=5,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        shuffle=False
    )

    train_transform = Compose([
        RandomResizedCrop(size=args.input_size, scale=(0.8, 1.)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = VOCDataset(root=args.data_path, image_set='trainaug', transforms=train_transform, return_masks=True)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    optimizer = torch.optim.SGD(
        seg_head.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 60., # linear scaling rule
        momentum=0.9,
        weight_decay=0.0001, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)    # decay rate default: 0.1

    to_restore = {"epoch": 0, "best_miou": 0.}
    start_epoch = to_restore["epoch"]
    best_miou = to_restore["best_miou"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, seg_head, optimizer, train_loader, epoch, args.spatial_size, args.train_mask_size, args.input_size)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, seg_head, args.num_classes, args.spatial_size, args.eval_size)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['Mean IoU']:.3f}")
            best_miou = max(best_miou, test_stats['Mean IoU'])
            print(f'Max MIoU so far: {best_miou:.3f}')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": seg_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_miou": best_miou,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test miou: {miou:.3f}".format(miou=best_miou))


def train(model, seg_head, optimizer, loader, epoch, spatial_size, train_mask_size, input_size, ignore_index=255):
    seg_head.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 5, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        B = inp.shape[0]

        # forward
        with torch.no_grad():
            output = model.backbone.forward(inp, None, None)[0]
            output = nn.functional.interpolate(output, size=(train_mask_size, train_mask_size), mode='bilinear')
        
        preds = seg_head(output)
        
        mask = target * 255
        if train_mask_size != input_size:
            with torch.no_grad():
                mask = nn.functional.interpolate(mask, size=(train_mask_size, train_mask_size), mode='nearest')
            
        # compute cross entropy loss
        loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(preds, mask.long().squeeze())

        # compute the gradients 
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, seg_head, num_classes, spatial_size, eval_size):
    seg_head.eval()
    metric = StreamSegMetrics(n_classes=num_classes)
    
    with torch.no_grad():
        for i, (inp, target) in enumerate(tqdm(val_loader)):
            # move to gpu
            inp = inp.cuda(non_blocking=True)

            # forward            
            output = model.backbone.forward(inp, None, None)[0]
            output = nn.functional.interpolate(output, size=(eval_size, eval_size), mode='bilinear')
            preds = seg_head(output)
            preds = torch.argmax(preds, dim=1).unsqueeze(1) # [B, 1, 448, 448]

            gt = target * 255   # [B, 1, 448, 448]
            gt = nn.functional.interpolate(gt, size=(eval_size, eval_size), mode='nearest')

            metric.update(label_trues=gt.numpy(), label_preds=preds.cpu().numpy())

    # calculate miou
    result = metric.get_results()
    print(result)
    metric.reset()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--epochs', default=25, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.01, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 60.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=60, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/dataset/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')

    parser.add_argument("--head_type", type=str, choices=['linear', 'fcn'], default="linear")
    parser.add_argument('--num_classes', default=21, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--eval_size', default=448, type=int, help='Size of groundtruth mask')
    parser.add_argument('--input_size', default=448, type=int, help='Size of input image')
    parser.add_argument('--train_mask_size', default=100, type=int, help='Size of gt mask used during training')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

    parser.add_argument('--layer_indices', default=-5, type=int)
    parser.add_argument('--num_cls_tokens', default=1, type=int)
    
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.spatial_size = args.input_size // args.patch_size
    eval_linear(args)
