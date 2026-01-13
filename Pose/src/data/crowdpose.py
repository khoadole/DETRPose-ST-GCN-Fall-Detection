"""
DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
Copyright (c) 2025 The DETRPose Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from GroupPose (https://github.com/Michel-liu/GroupPose/)
Copyright (c) 2023 GroupPose Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from ED-Pose (https://github.com/IDEA-Research/ED-Pose/)
Copyright (c) 2023 IDEA. All Rights Reserved.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from xtcocotools.coco import COCO

class CrowdPoseDetection(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms, return_masks=False):
        super(CrowdPoseDetection, self).__init__()
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        
        self.img_folder = Path(img_folder)
        self.coco = COCO(ann_file)
        imgIds = sorted(self.coco.getImgIds())
        
        if "train" in ann_file:
            self.all_imgIds = []
            for image_id in imgIds:
                if self.coco.getAnnIds(imgIds=image_id) == []:
                    continue
                ann_ids = self.coco.getAnnIds(imgIds=image_id)
                target = self.coco.loadAnns(ann_ids)
                num_keypoints = [obj["num_keypoints"] for obj in target]
                if sum(num_keypoints) == 0:
                    continue
                self.all_imgIds.append(image_id)
        else:
            self.all_imgIds = []
            for image_id in imgIds:
                self.all_imgIds.append(image_id)

    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __len__(self):
        return len(self.all_imgIds)

    def load_item(self, idx):
        image_id = self.all_imgIds[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        target = self.coco.loadAnns(ann_ids)

        target = {'image_id': image_id, 'annotations': target}
        img = Image.open(self.img_folder / self.coco.loadImgs(image_id)[0]['file_name'])
        img, target = self.prepare(img, target)
        return img, target

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target = self._transforms(img, target, self)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(img_array)
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        anno = [obj for obj in anno if obj['num_keypoints'] != 0]
        keypoints = [obj["keypoints"] for obj in anno]
        boxes = [obj["bbox"] for obj in anno]
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, 14, 3)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        keypoints = keypoints[keep]
        if self.return_masks:
            masks = masks[keep]
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target

