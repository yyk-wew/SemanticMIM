import random
import torchvision.transforms as T
import torchvision.transforms.functional as F

from typing import Optional, Callable, List, Tuple, Dict
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from typing import Tuple, Any
import os
import sys

from torchmetrics import Metric
import torch
from collections import defaultdict
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.hflip(img)
            target = F.hflip(target)
        return img, target


class RandomResizedCrop(object):
    def __init__(self, size, scale, ratio=(3. / 4., 4. / 3.)):
        self.rrc_transform = T.RandomResizedCrop(size=size, scale=scale, ratio=ratio)

    def __call__(self, img, target=None):
        y1, x1, h, w = self.rrc_transform.get_params(img, self.rrc_transform.scale, self.rrc_transform.ratio)
        img = F.resized_crop(img, y1, x1, h, w, self.rrc_transform.size, F.InterpolationMode.BILINEAR)
        target = F.resized_crop(target, y1, x1, h, w, self.rrc_transform.size, F.InterpolationMode.NEAREST)
        return img, target


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), F.to_tensor(target)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class VOCDataset(VisionDataset):

    def __init__(
            self,
            root: str,
            image_set: str = "trainaug",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            return_masks: bool = False
    ):
        super(VOCDataset, self).__init__(root, transforms, transform, target_transform)
        self.image_set = image_set
        if self.image_set == "trainaug" or self.image_set == "train":
            seg_folder = "SegmentationClassAug"
        elif self.image_set == "val":
            seg_folder = "SegmentationClass"
        else:
            raise ValueError(f"No support for image set {self.image_set}")
        seg_dir = os.path.join(root, seg_folder)
        image_dir = os.path.join(root, 'JPEGImages')
        if not os.path.isdir(seg_dir) or not os.path.isdir(image_dir) or not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.')
        splits_dir = os.path.join(root, 'ImageSets', 'Segmentation')
        split_f = os.path.join(splits_dir, self.image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(seg_dir, x + ".png") for x in file_names]

        # dummy implementation (Should combine SegmentationClass and SegmentationClassAug)
        for i in range(len(self.masks)):
            if not Path(self.masks[i]).is_file():
                self.masks[i] = os.path.join(root, "SegmentationClass", os.path.split(self.masks[i])[1])

        # print(split_f, self.image_set, len(self.masks), len(self.images))

        self.return_masks = return_masks

        assert all([Path(f).is_file() for f in self.masks]) and all([Path(f).is_file() for f in self.images])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        if self.image_set == "val":
            mask = Image.open(self.masks[index])
            if self.transforms:
                img, mask = self.transforms(img, mask)
            return img, mask
        elif "train" in self.image_set:
            if self.transforms:
                if self.return_masks:
                    mask = Image.open(self.masks[index])
                    res = self.transforms(img, mask)
                else:
                    res = self.transforms(img)
                return res
            return img

    def __len__(self) -> int:
        return len(self.images)


class PredsmIoU(Metric):
    """
    Subclasses Metric. Computes mean Intersection over Union (mIoU) given ground-truth and predictions.
    .update() can be called repeatedly to add data from multiple validation loops.
    """
    def __init__(self,
                 num_pred_classes: int,
                 num_gt_classes: int):
        """
        :param num_pred_classes: The number of predicted classes.
        :param num_gt_classes: The number of gt classes.
        """
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.num_pred_classes = num_pred_classes
        self.num_gt_classes = num_gt_classes
        self.add_state("gt", [])
        self.add_state("pred", [])
        self.n_jobs = -1

    def update(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        self.gt.append(gt)
        self.pred.append(pred)

    def compute(self, is_global_zero: bool, many_to_one: bool = False,
                precision_based: bool = False, linear_probe : bool = False) -> Tuple[float, List[np.int64],
                                                                                     List[np.int64], List[np.int64],
                                                                                     List[np.int64], float]:
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param is_global_zero: Flag indicating whether process is rank zero. Computation of metric is only triggered
        if True.
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt,  percentage of clusters matched to background class. 1/self.num_pred_classes
        if self.num_pred_classes == self.num_gt_classes.
        """
        if is_global_zero:
            pred = torch.cat(self.pred).cpu().numpy().astype(int)
            gt = torch.cat(self.gt).cpu().numpy().astype(int)
            assert len(np.unique(pred)) <= self.num_pred_classes
            assert np.max(pred) <= self.num_pred_classes
            return self.compute_miou(gt, pred, self.num_pred_classes, self.num_gt_classes, many_to_one=many_to_one,
                                     precision_based=precision_based, linear_probe=linear_probe)

    def compute_miou(self, gt: np.ndarray, pred: np.ndarray, num_pred: int, num_gt:int,
                     many_to_one=False, precision_based=False, linear_probe=False) -> Tuple[float, List[np.int64], List[np.int64], List[np.int64],
                                                  List[np.int64], float]:
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param gt: numpy array with all flattened ground-truth class assignments per pixel
        :param pred: numpy array with all flattened class assignment predictions per pixel
        :param num_pred: number of predicted classes
        :param num_gt: number of ground truth classes
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt,  percentage of clusters matched to background class. 1/self.num_pred_classes
        if self.num_pred_classes == self.num_gt_classes.
        """
        assert pred.shape == gt.shape
        print(f"seg map preds have size {gt.shape}")
        tp = [0] * num_gt
        fp = [0] * num_gt
        fn = [0] * num_gt
        jac = [0] * num_gt

        reordered_preds = pred

        # tp, fp, and fn evaluation
        for i_part in range(0, num_gt):
            tmp_all_gt = (gt == i_part)
            tmp_pred = (reordered_preds == i_part)
            tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
            fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
            fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)

        # Calculate IoU per class
        for i_part in range(0, num_gt):
            jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

        print("IoUs computed")
        return np.mean(jac), tp, fp, fn, reordered_preds.astype(int).tolist()


class StreamSegMetrics(object):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.total_samples = 0

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        self.total_samples += len(label_trues)

    def to_str(self, results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU" and k != "Class Acc" and k != "Confusion Matrix":
                string += "%s: %f\n" % (k, v)

        string += 'Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += 'Class Acc:\n'
        for k, v in results['Class Acc'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes**2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        EPS = 1e-6
        hist = self.confusion_matrix

        gt_sum = hist.sum(axis=1)
        mask = (gt_sum != 0)
        diag = np.diag(hist)

        acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + EPS)
        acc_cls = np.mean(acc_cls_c[mask])
        iu = diag / (gt_sum + hist.sum(axis=0) - diag + EPS)
        # mean_iu = np.mean(iu[mask])
        mean_iu = np.mean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), [iu[i] if m else "X" for i, m in enumerate(mask)]))
        cls_acc = dict(
            zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)])
        )

        return {
            # "Total samples": self.total_samples,
            # "Overall Acc": acc,
            # "Mean Acc": acc_cls,
            # "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            # "Class IoU": cls_iu,
            # "Class Acc": cls_acc,
            # "Confusion Matrix": self.confusion_matrix_to_fig()
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_samples = 0

    def synch(self, device):
        # collect from multi-processes
        confusion_matrix = torch.tensor(self.confusion_matrix).to(device)
        samples = torch.tensor(self.total_samples).to(device)

        torch.distributed.reduce(confusion_matrix, dst=0)
        torch.distributed.reduce(samples, dst=0)

        if torch.distributed.get_rank() == 0:
            self.confusion_matrix = confusion_matrix.cpu().numpy()
            self.total_samples = samples.cpu().numpy()

    def confusion_matrix_to_fig(self):
        cm = self.confusion_matrix.astype('float') / (self.confusion_matrix.sum(axis=1) +
                                                      0.000001)[:, np.newaxis]
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(title=f'Confusion Matrix', ylabel='True label', xlabel='Predicted label')

        fig.tight_layout()
        return fig