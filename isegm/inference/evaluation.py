from time import time

import numpy as np
import torch
import cv2

from isegm.inference import utils
from isegm.inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def _apply_postprocessing_by_edge_gray(image_rgb, original_mask, edge_threshold=0.5):
    """基于前景平均灰度与外边缘灰度比值的简单后处理扩展。
    逻辑与交互式控制器中的实现保持一致：
    - 计算原始前景掩码的平均灰度
    - 对前景掩码进行一次膨胀，得到紧邻前景的一圈背景像素（outer_border）
    - 若外边缘像素灰度 / 前景平均灰度 > edge_threshold，则将该像素并入前景
    """
    if image_rgb is None or original_mask is None:
        return original_mask

    # 计算前景平均灰度
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    foreground_pixels = gray_image[original_mask]

    extended_mask = original_mask.copy()
    if foreground_pixels.size == 0:
        return extended_mask

    avg_gray_value = float(np.mean(foreground_pixels))
    if avg_gray_value <= 0:
        return extended_mask

    # 计算外边缘
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(original_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    outer_border = np.logical_and(dilated, np.logical_not(original_mask))

    by, bx = np.where(outer_border)
    if by.size > 0:
        border_gray = gray_image[by, bx].astype(np.float32)
        ratio = border_gray / avg_gray_value
        take = ratio > float(edge_threshold)
        if np.any(take):
            extended_mask[by[take], bx[take]] = True

    return extended_mask


def _enforce_click_constraints(pred_mask, clicks_list):
    """根据点击类型强制掩码满足期望：
    - 正样本点击点必须落在前景掩码中
    - 负样本点击点必须落在背景掩码中
    """
    if pred_mask is None or clicks_list is None:
        return pred_mask

    h, w = pred_mask.shape
    for click in clicks_list:
        y, x = click.coords
        y, x = int(round(y)), int(round(x))
        if 0 <= y < h and 0 <= x < w:
            if click.is_positive:
                pred_mask[y, x] = True
            else:
                pred_mask[y, x] = False
    return pred_mask


def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        for object_id in sample.objects_ids:
            _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask(object_id), predictor,
                                                sample_id=index, **kwargs)
            all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None,
                    postproc_enabled=False, edge_threshold=0.5):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            # 初始掩码
            pred_mask = pred_probs > pred_thr

            # 强制执行点击约束，保证正负点击点达到预期效果，放在后处理前
            pred_mask = _enforce_click_constraints(pred_mask, clicker.clicks_list)

            # 可选后处理：根据图像灰度边缘扩展前景掩码
            if postproc_enabled:
                try:
                    pred_mask = _apply_postprocessing_by_edge_gray(image, pred_mask, edge_threshold=edge_threshold)
                except Exception:
                    # 保底容错：若后处理失败，回退到原始掩码
                    pred_mask = pred_probs > pred_thr

            # 强制执行点击约束，保证正负点击点达到预期效果，放在后处理前
            pred_mask = _enforce_click_constraints(pred_mask, clicker.clicks_list)

            if callback is not None:
                # 可视化使用概率图与强制后的掩码
                callback(image, gt_mask, pred_probs, pred_mask, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
