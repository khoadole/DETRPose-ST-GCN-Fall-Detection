from src.core import LazyCall as L
from src.data import CrowdPoseDetection
from src.data.dataloader import (
	BatchImageCollateFunction, 
	DataLoader
	)
from src.data.crowdpose_eval import CrowdPoseEvaluator
from src.data.container import Compose
import src.data.transforms as T
import src.data.transforms_crowdpose as CrowdT

from .detrpose_hgnetv2 import eval_spatial_size

from omegaconf import OmegaConf

scales = [(640, 640)]
max_size = 1333
scales2_resize = [400, 500, 600]

__all__ = ["dataset_train", "dataset_val", "dataset_test", "evaluator"]

dataset_train = L(DataLoader)(
	dataset=L(CrowdPoseDetection)(
		img_folder="./data/crowdpose/images",
		ann_file="./data/crowdpose/annotations/crowdpose_trainval.json",
		transforms=L(Compose)(
			policy={
				'name': 'stop_epoch',
				'ops': ['Mosaic', 'RandomCrop', 'RandomZoomOut'],
				'epoch': [5, 29, 60]
				},
			mosaic_prob=0.5,
			transforms1=L(T.Mosaic)(output_size=320, probability=1.0),
			transforms2=L(T.RandomZoomOut)(p=0.5),
			transforms3=L(CrowdT.RandomHorizontalFlip)(p=0.5),
			transforms4=L(T.ColorJitter)(),
			transforms5=L(T.RandomResize)(sizes=scales, max_size=max_size), 
			transforms6=L(T.ToTensor)(),
			transforms7=L(T.Normalize)(mean=[0, 0, 0], std=[1, 1, 1])	
			),

		),
	total_batch_size=16,
	collate_fn=L(BatchImageCollateFunction)(
		base_size=eval_spatial_size[0],
		base_size_repeat=4,
		stop_epoch=60,
		),
	num_workers=4,
	shuffle=True,
	drop_last=True,
	pin_memory=True
	)

dataset_val = L(DataLoader)(
	dataset=L(CrowdPoseDetection)(
		img_folder="./data/crowdpose/images",
		ann_file="./data/crowdpose/annotations/crowdpose_test.json",
		transforms=L(Compose)(
			transforms1=L(T.RandomResize)(sizes=[eval_spatial_size], max_size=max_size), 
			transforms2=L(T.ToTensor)(),
			transforms3=L(T.Normalize)(mean=[0, 0, 0], std=[1, 1, 1])
			),
		),
	total_batch_size=32,
	collate_fn=L(BatchImageCollateFunction)(
		base_size=eval_spatial_size[0],
		),
	num_workers=4,
	shuffle=False,
	drop_last=False,
	pin_memory=True
	)

dataset_test = L(DataLoader)(
	dataset=L(CrowdPoseDetection)(
		img_folder="./data/crowdpose/images",
		ann_file="./data/crowdpose/annotations/crowdpose_test.json",
		transforms=L(Compose)(
			transforms1=L(T.RandomResize)(sizes=[eval_spatial_size], max_size=max_size), 
			transforms2=L(T.ToTensor)(),
			transforms3=L(T.Normalize)(mean=[0, 0, 0], std=[1, 1, 1])
			),
		),
	total_batch_size=32,
	collate_fn=L(BatchImageCollateFunction)(
		base_size=eval_spatial_size[0],
		),
	num_workers=4,
	shuffle=False,
	drop_last=False,
	pin_memory=True
	)

evaluator = L(CrowdPoseEvaluator)(
	ann_file="./data/crowdpose/annotations/crowdpose_test.json",
	iou_types=['keypoints_crowd'],
	useCats=True
	)

