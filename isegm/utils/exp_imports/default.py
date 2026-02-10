import torch
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from isegm.data.datasets import *
from isegm.data.datasets.SIRST3 import *
from isegm.data.datasets.WideIRSTD import *
from isegm.model.losses import *
from isegm.data.transforms import *
from isegm.engine.trainer import ISTrainer
from isegm.engine.trainer_segnext import ISTrainer_segnext
from isegm.model.metrics import AdaptiveIoU
from isegm.data.points_sampler import MultiPointSampler
from isegm.utils.log import logger
from isegm.model import initializer

from isegm.model.is_hrnet_model import HRNetModel
from isegm.model.is_deeplab_model import DeeplabModel
from isegm.model.is_segformer_model import SegFormerModel
from isegm.model.is_hrformer_model import HRFormerModel
from isegm.model.is_swinformer_model import SwinformerModel
from isegm.model.is_plainvit_model import PlainVitModel
from isegm.model.is_plainvit_model_segnext import PlainVitModel_segnext
from isegm.model.is_plainvit_model_SIRST3 import PlainVitModel_SIRST3
from isegm.model.is_plainvit_model_WideIRSTD import PlainVitModel_WideIRSTD