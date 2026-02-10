from datetime import timedelta
from pathlib import Path
import torch
import numpy as np

from isegm.data.datasets.SIRST3 import SIRST3Dataset
from isegm.data.datasets.WideIRSTD import WideIRSTD
from isegm.utils.serialization import load_model


def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def load_is_model(checkpoint, device, eval_ritm, **kwargs):
    if isinstance(checkpoint, (str, Path)):
        state_dict = torch.load(checkpoint, map_location='cpu')
        # print("Load pre-trained checkpoint from: %s" % checkpoint)
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0], device, eval_ritm, **kwargs)
        models = [load_single_is_model(x, device, eval_ritm, **kwargs) for x in state_dict]

        return model, models
    else:
        return load_single_is_model(state_dict, device, eval_ritm, **kwargs)


def load_single_is_model(state_dict, device, eval_ritm, **kwargs):
    # 根据 checkpoint 中是否包含 maps_transform 权重，自动对齐 use_rgb_conv 以避免键不匹配
    sd = state_dict['state_dict']
    has_maps_transform = any(k.startswith('maps_transform.') for k in sd.keys())

    model = load_model(state_dict['config'], eval_ritm, use_rgb_conv=has_maps_transform, **kwargs)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        # 严格加载失败（常见于不同版本/配置的轻微不一致），回退为非严格加载
        model.load_state_dict(sd, strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model


def get_dataset(dataset_name, cfg):
    if dataset_name == 'SIRST3':
        dataset = SIRST3Dataset(cfg.SIRST3_PATH, split='test')
    elif dataset_name == 'WideIRSTD':
        dataset = WideIRSTD(cfg.WideIRSTD_PATH, split='test')
    else:
        dataset = None

    return dataset


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    noc_list_std = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array([_get_noc(iou_arr, iou_thr)
                               for iou_arr in all_ious], dtype=int)

        score = scores_arr.mean()
        score_std = scores_arr.std()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        noc_list_std.append(score_std)
        over_max_list.append(over_max)

    return noc_list, noc_list_std, over_max_list


def find_checkpoint(weights_folder, checkpoint_name):
    weights_folder = Path(weights_folder)
    if ':' in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(':')
        models_candidates = [x for x in weights_folder.glob(f'{model_name}*') if x.is_dir()]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith('.pth'):
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f'{checkpoint_name}*.pth'))
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]

    return str(checkpoint_path)


def get_results_table(noc_list, over_max_list, brs_type, dataset_name, mean_spc, elapsed_time,
                      n_clicks=20, model_name=None, first_click_miou=None):
    table_header = (f'|{"BRS Type":^13}|{"Dataset":^11}|'
                    f'{"1-mIoU":^9}|'
                    f'{"NoC@70%":^9}|{"NoC@80%":^9}|{"NoC@90%":^9}|'
                    f'{">="+str(n_clicks)+"@70%":^9}|{">="+str(n_clicks)+"@80%":^9}|{">="+str(n_clicks)+"@90%":^9}|'
                    f'{"SPC,s":^7}|{"Time":^9}|')
    row_width = len(table_header)

    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f'|{brs_type:^13}|{dataset_name:^11}|'
    # 1-mIoU 以百分比显示
    if first_click_miou is not None:
        table_row += f'{first_click_miou:^9.2%}|'
    else:
        table_row += f'{"?":^9}|'
    table_row += f'{noc_list[0]:^9.2f}|'
    table_row += f'{noc_list[1]:^9.2f}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{noc_list[2]:^9.2f}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{over_max_list[0]:^9}|' if len(over_max_list) > 0 else f'{"?":^9}|'
    table_row += f'{over_max_list[1]:^9}|' if len(over_max_list) > 1 else f'{"?":^9}|'
    table_row += f'{over_max_list[2]:^9}|' if len(over_max_list) > 2 else f'{"?":^9}|'
    table_row += f'{mean_spc:^7.3f}|{eval_time:^9}|'

    return header, table_row
