import random
import pickle
import numpy as np
import torch
from torchvision import transforms
from .points_sampler import MultiPointSampler
from .sample import DSample


class ISDataset(torch.utils.data.dataset.Dataset):
    def __init__(self,
                 augmentator=None,
                 points_sampler=MultiPointSampler(max_num_points=12),
                 min_object_area=0,
                 keep_background_prob=0.0,
                 with_image_info=False,
                 samples_scores_path=None,
                 samples_scores_gamma=1.0,
                 epoch_len=-1,
                 sampling_phase=1):
        super(ISDataset, self).__init__()
        self.epoch_len = epoch_len
        self.augmentator = augmentator
        self.min_object_area = min_object_area
        self.keep_background_prob = keep_background_prob
        self.points_sampler = points_sampler
        self.with_image_info = with_image_info
        # Store sampling configuration
        self.samples_scores_path = samples_scores_path
        self.samples_scores_gamma = samples_scores_gamma
        self.sampling_phase = sampling_phase

        # Load precomputed sample scores and initialize probs for the selected phase
        self.samples_precomputed_scores = self._load_samples_scores(samples_scores_path, samples_scores_gamma)
        if self.samples_precomputed_scores is not None:
            # Default to phase 1 unless explicitly set otherwise
            if self.sampling_phase == 2:
                self.samples_precomputed_scores['probs'] = self.samples_precomputed_scores['probs_stage2']
            else:
                self.samples_precomputed_scores['probs'] = self.samples_precomputed_scores['probs_stage1']
        self.to_tensor = transforms.ToTensor()

        self.dataset_samples = None

    def __getitem__(self, index):
        if self.samples_precomputed_scores is not None:
            index = np.random.choice(self.samples_precomputed_scores['indices'],
                                     p=self.samples_precomputed_scores['probs'])
        else:
            if self.epoch_len > 0:
                index = random.randrange(0, len(self.dataset_samples))

        sample = self.get_sample(index)
        sample = self.augment_sample(sample)
        sample.remove_small_objects(self.min_object_area)

        self.points_sampler.sample_object(sample)
        points = np.array(self.points_sampler.sample_points())
        mask = self.points_sampler.selected_mask

        output = {
            'images': self.to_tensor(sample.image),
            'points': points.astype(np.float32),
            'instances': mask
        }

        if self.with_image_info:
            output['image_info'] = sample.sample_id

        return output

    def augment_sample(self, sample) -> DSample:
        if self.augmentator is None:
            return sample

        valid_augmentation = False
        while not valid_augmentation:
            sample.augment(self.augmentator)
            keep_sample = (self.keep_background_prob < 0.0 or
                           random.random() < self.keep_background_prob)
            valid_augmentation = len(sample) > 0 or keep_sample

        return sample

    def get_sample(self, index) -> DSample:
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        else:
            return self.get_samples_number()

    def get_samples_number(self):
        return len(self.dataset_samples)

    @staticmethod
    def _load_samples_scores(samples_scores_path, samples_scores_gamma):
        if samples_scores_path is None:
            return None

        with open(samples_scores_path, 'rb') as f:
            images_scores = pickle.load(f)

        # 第一阶段训练（偏向容易&中等难度样本）：公式1
        probs_stage1 = np.array([(1.0 - x[2]) ** samples_scores_gamma for x in images_scores])
        probs_stage1 /= probs_stage1.sum()

        # 第二阶段训练（偏向困难样本）：公式2
        probs_stage2 = np.array([x[2] ** samples_scores_gamma for x in images_scores])
        probs_stage2 /= probs_stage2.sum()
        samples_scores = {
            'indices': [x[0] for x in images_scores],
            # 默认不直接设置 'probs'，由初始化或显式切换阶段决定
            'probs_stage1': probs_stage1,
            'probs_stage2': probs_stage2
        }
        print(f'Loaded {len(images_scores)} weights with gamma={samples_scores_gamma}')
        return samples_scores

    def set_sampling_phase(self, phase: int):
        """切换采样阶段：
        phase=1 使用公式1（偏向容易&中等难度样本）；
        phase=2 使用公式2（偏向困难样本）。
        """
        if self.samples_precomputed_scores is None:
            return
        if phase == 2:
            self.samples_precomputed_scores['probs'] = self.samples_precomputed_scores['probs_stage2']
            self.sampling_phase = 2
            print('Switched dataset sampling to Phase 2 (hard-focused).')
        else:
            self.samples_precomputed_scores['probs'] = self.samples_precomputed_scores['probs_stage1']
            self.sampling_phase = 1
            print('Switched dataset sampling to Phase 1 (easy/medium-focused).')
