import pickle as pkl
from pathlib import Path
import os
import cv2
import numpy as np
from scipy.io import loadmat

from isegm.utils.misc import get_bbox_from_mask, get_labels_with_sizes
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class SIRST3Dataset(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super(SIRST3Dataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'images'
        self._masks_path = self.dataset_path / 'masks_inst'
        # self._buggy_objects = dict()
        # self._buggy_mask_thresh = buggy_mask_thresh

        with open(os.path.join(self.dataset_path, 'img_idx', f'{split}.txt'), 'r') as f:
            self.dataset_samples = [x.strip() for x in f.readlines()] # x.strip()返回一个去除字符串 x 首尾空白字符（包括空格、换行符 \n、制表符 \t 等）的新字符串

    def get_sample(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.png')
        mask_info_path = str(self._masks_path / f'{image_name}.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_info_path, cv2.IMREAD_GRAYSCALE)
        # instances_mask = self.remove_buggy_masks(index, instances_mask)
        instances_ids, _ = get_labels_with_sizes(instances_mask)

        return DSample(image, instances_mask, objects_ids=instances_ids, sample_id=index)
    #
    # def remove_buggy_masks(self, index, instances_mask):
    #     if self._buggy_mask_thresh > 0.0:
    #         buggy_image_objects = self._buggy_objects.get(index, None)
    #         if buggy_image_objects is None:
    #             buggy_image_objects = []
    #             instances_ids, _ = get_labels_with_sizes(instances_mask)
    #             for obj_id in instances_ids:
    #                 obj_mask = instances_mask == obj_id
    #                 mask_area = obj_mask.sum()
    #                 bbox = get_bbox_from_mask(obj_mask)
    #                 bbox_area = (bbox[1] - bbox[0] + 1) * (bbox[3] - bbox[2] + 1)
    #                 obj_area_ratio = mask_area / bbox_area
    #                 if obj_area_ratio < self._buggy_mask_thresh:
    #                     buggy_image_objects.append(obj_id)
    #
    #             self._buggy_objects[index] = buggy_image_objects
    #         for obj_id in buggy_image_objects:
    #             instances_mask[instances_mask == obj_id] = 0
    #
    #     return instances_mask


# class SIRST3_ValDataset(ISDataset):
#     def __init__(self, dataset_path, split='val', **kwargs):
#         super(SIRST3_ValDataset, self).__init__(**kwargs)
#         assert split in {'train', 'val'}
#
#         self.dataset_path = Path(dataset_path)
#         self.dataset_split = split
#         self._images_path = self.dataset_path / 'images'
#         self._masks_path = self.dataset_path / 'instances'
#
#         with open(self.dataset_path / f'{split}.txt', 'r') as f:
#             self.dataset_samples = [x.strip() for x in f.readlines()]
#
#         self.dataset_samples = self.get_sbd_images_and_ids_list()
#
#     def get_sample(self, index) -> DSample:
#         image_name, instance_id = self.dataset_samples[index]
#         image_path = str(self._images_path / f'{image_name}.jpg')
#         inst_info_path = str(self._insts_path / f'{image_name}.mat')
#
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         instances_mask = loadmat(str(inst_info_path))['GTinst'][0][0][0].astype(np.int32)
#         instances_mask[instances_mask != instance_id] = 0
#         instances_mask[instances_mask > 0] = 1
#
#         return DSample(image, instances_mask, objects_ids=[1], sample_id=index)
#
#     def get_sbd_images_and_ids_list(self):
#         pkl_path = self.dataset_path / f'{self.dataset_split}_images_and_ids_list.pkl'
#
#         if pkl_path.exists():
#             with open(str(pkl_path), 'rb') as fp:
#                 images_and_ids_list = pkl.load(fp)
#         else:
#             images_and_ids_list = []
#
#             for sample in self.dataset_samples:
#                 inst_info_path = str(self._insts_path / f'{sample}.mat')
#                 instances_mask = loadmat(str(inst_info_path))['GTinst'][0][0][0].astype(np.int32)
#                 instances_ids, _ = get_labels_with_sizes(instances_mask)
#
#                 for instances_id in instances_ids:
#                     images_and_ids_list.append((sample, instances_id))
#
#             with open(str(pkl_path), 'wb') as fp:
#                 pkl.dump(images_and_ids_list, fp)
#
#         return images_and_ids_list