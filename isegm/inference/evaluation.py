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
    """Simple post-processing extension based on the ratio of foreground average gray value to outer edge gray value.
    The logic is consistent with the implementation in the interactive controller:
    - Calculate the average gray value of the original foreground mask
    - Dilate the foreground mask once to get a ring of background pixels immediately adjacent to the foreground (outer_border)
    - If outer edge pixel gray value / foreground average gray value > edge_threshold, merge that pixel into the foreground
    """
    if image_rgb is None or original_mask is None:
        return original_mask

    # Calculate foreground average gray value
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    foreground_pixels = gray_image[original_mask]

    extended_mask = original_mask.copy()
    if foreground_pixels.size == 0:
        return extended_mask

    avg_gray_value = float(np.mean(foreground_pixels))
    if avg_gray_value <= 0:
        return extended_mask

    # Calculate outer edge
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
    """Enforce mask constraints based on click types:
    - Positive clicks must fall within the foreground mask
    - Negative clicks must fall within the background mask
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
            # Initial mask
            pred_mask = pred_probs > pred_thr

            # Enforce click constraints to ensure positive/negative clicks achieve expected effects, placed before post-processing
            pred_mask = _enforce_click_constraints(pred_mask, clicker.clicks_list)

            # Optional post-processing: Extend foreground mask based on image gray edges
            if postproc_enabled:
                try:
                    pred_mask = _apply_postprocessing_by_edge_gray(image, pred_mask, edge_threshold=edge_threshold)
                except Exception:
                    # Fallback: If post-processing fails, revert to the original mask
                    pred_mask = pred_probs > pred_thr

            # Enforce click constraints to ensure positive/negative clicks achieve expected effects, placed after post-processing
            pred_mask = _enforce_click_constraints(pred_mask, clicker.clicks_list)

            if callback is not None:
                # Visualization uses probability map and enforced mask
                callback(image, gt_mask, pred_probs, pred_mask, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
