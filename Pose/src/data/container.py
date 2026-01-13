"""
DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
Copyright (c) 2025 The DETRPose Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-DEIM (https://github.com/Intellindust-AI-Lab/DEIM/)
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR/)
Copyright (c) 2023 RT-DETR Authors. All Rights Reserved.
"""

from omegaconf import ListConfig
import random

class Compose(object):
    def __init__(self, policy=None, mosaic_prob=0.0, **transforms):
        self.transforms = []
        for transform in transforms.values():
            self.transforms.append(transform)

        self.mosaic_prob = mosaic_prob

        if policy is None:
            self.policy = {'name': 'default'}
        else:
            self.policy = policy
            if self.mosaic_prob > 0:
                print("     ### Mosaic with Prob.@{} and RandomZoomOut/RandomCrop  existed ### ".format(self.mosaic_prob))
            print("     ### ImgTransforms Epochs: {} ### ".format(policy['epoch']))
            print('     ### Policy_ops@{} ###'.format(policy['ops']))

        ### warnings ##
        self.warning_mosaic_start = True

    def __call__(self, image, target, dataset=None):
        return self.get_forward(self.policy['name'])(image, target, dataset)

    def get_forward(self, name):
        forwards = {
            'default': self.default_forward,
            'stop_epoch': self.stop_epoch_forward,
        }
        return forwards[name]

    def default_forward(self, image, target, dataset=None):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

    def stop_epoch_forward(self, image, target, dataset=None):
        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_epoch = self.policy['epoch']

        if isinstance(policy_epoch, (list, ListConfig)) and len(policy_epoch) == 3:
            if policy_epoch[0] <= cur_epoch < policy_epoch[1]:
                with_mosaic = random.random() <= self.mosaic_prob       # Probility for Mosaic
            else:
                with_mosaic = False

            for transform in self.transforms:
                if (type(transform).__name__ in policy_ops and cur_epoch < policy_epoch[0]):   # first stage: NoAug
                    pass
                elif (type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch[-1]):   # last stage: NoAug
                    pass
                else:
                    # Using Mosaic for [policy_epoch[0], policy_epoch[1]] with probability
                    if (type(transform).__name__ == 'Mosaic' and not with_mosaic):      
                        pass
                    # Mosaic and Zoomout/IoUCrop can not be co-existed in the same sample
                    elif (type(transform).__name__ == 'RandomZoomOut' or type(transform).__name__ == 'RandomCrop') and with_mosaic:      
                        pass
                    else:
                        if type(transform).__name__ == 'Mosaic':
                            if self.warning_mosaic_start:
                                # It shows in which epochs mosaic is being used
                                print(f'     ### Mosaic is being used @ epoch {cur_epoch}...')
                                self.warning_mosaic_start = False
                            image, target = transform(image, target, dataset)
                        else:
                            image, target = transform(image, target)
        else:
            for transform in self.transforms:
                image, target = transform(image, target)

        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string