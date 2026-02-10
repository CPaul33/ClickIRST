import torch
import numpy as np
from tkinter import messagebox
import cv2

from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks


class InteractiveController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.5):
        self.net = net
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None
        self._postprocessed_mask = None  # 存储后处理后的掩码
        self._extended_pixels = None  # 存储新增的前景像素
        self.enable_postprocessing = False  # 后处理功能开关
        self.edge_threshold = 0.5  # 外边缘灰度值阈值

        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()

    def _enforce_clicks_on_bool_mask(self, mask):
        """强制点击约束到布尔掩码：
        - 正样本点赋值为 True；负样本点赋值为 False
        """
        if mask is None:
            return mask
        h, w = mask.shape
        for click in self.clicker.clicks_list:
            y, x = click.coords
            y, x = int(round(y)), int(round(x))
            if 0 <= y < h and 0 <= x < w:
                mask[y, x] = True if click.is_positive else False
        return mask

    def _enforce_clicks_on_label_mask(self, mask, fg_value, bg_value=0):
        """强制点击约束到标签掩码：
        - 正样本点赋值为前景标签；负样本点赋值为背景标签
        """
        if mask is None:
            return mask
        h, w = mask.shape
        for click in self.clicker.clicks_list:
            y, x = click.coords
            y, x = int(round(y)), int(round(x))
            if 0 <= y < h and 0 <= x < w:
                mask[y, x] = fg_value if click.is_positive else bg_value
        return mask

    def set_postprocessing_enabled(self, enabled):
        self.enable_postprocessing = enabled
        if self.image is not None:
            self._update_postprocessed_mask()
            self.update_image_callback()

    def set_edge_threshold(self, threshold):
        self.edge_threshold = threshold
        if self.enable_postprocessing and self.image is not None:
            self._update_postprocessed_mask()
            self.update_image_callback()

    def set_prob_thresh(self, prob_thresh):
        self.prob_thresh = float(prob_thresh)
        if self.image is not None:
            # 概率阈值变化会影响原始掩码与平均灰度统计，需重新计算后处理结果
            self._update_postprocessed_mask()
            self.update_image_callback()

    def _update_postprocessed_mask(self):
        """更新后处理掩码"""
        if not self.enable_postprocessing or self.image is None:
            self._postprocessed_mask = None
            self._extended_pixels = None
            return

        current_prob = self.current_object_prob
        if current_prob is None:
            self._postprocessed_mask = None
            self._extended_pixels = None
            return

        # 获取原始掩码，并在后处理之前先施加点击强约束
        original_mask = current_prob > self.prob_thresh
        original_mask = self._enforce_clicks_on_bool_mask(original_mask)

        # 计算前景区域的平均灰度值
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        foreground_pixels = gray_image[original_mask]
        if len(foreground_pixels) == 0:
            self._postprocessed_mask = original_mask
            self._extended_pixels = None
            return

        avg_gray_value = np.mean(foreground_pixels)

        # 计算“外边缘”：对前景做一次膨胀，取膨胀结果与原掩码的差集，得到紧邻前景的一圈背景像素
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(original_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        outer_border = np.logical_and(dilated, np.logical_not(original_mask))

        # 初始化扩展掩码与新增像素标记
        extended_mask = original_mask.copy()
        extended_pixels = np.zeros_like(original_mask, dtype=bool)

        # 向量化判断边缘像素是否需要纳入前景
        if avg_gray_value > 0:
            by, bx = np.where(outer_border)
            if by.size > 0:
                border_gray = gray_image[by, bx].astype(np.float32)
                ratio = border_gray / float(avg_gray_value)
                take = ratio > float(self.edge_threshold)
                if np.any(take):
                    extended_mask[by[take], bx[take]] = True
                    extended_pixels[by[take], bx[take]] = True
                    print(f"后处理算法扩展了 {int(np.count_nonzero(take))} 个像素")
                else:
                    print("后处理算法扩展了 0 个像素")
            else:
                print("后处理算法扩展了 0 个像素")
        else:
            # 极端情况：前景平均灰度为0，无法比较，保持原样
            print("前景平均灰度为0，未进行边缘扩展")

        # 在后处理之后再次施加点击强约束，防止扩展覆盖负点或清除正点
        self._postprocessed_mask = self._enforce_clicks_on_bool_mask(extended_mask)
        self._extended_pixels = extended_pixels

    def set_image(self, image):
        self.image = image
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self._update_postprocessed_mask()
        self.update_image_callback(reset_canvas=True)

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        self.clicker.click_indx_offset = 1
        self._update_postprocessed_mask()

    def add_click(self, x, y, is_positive):
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        if self._init_mask is not None and len(self.clicker) == 1:
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)

        torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        self._update_postprocessed_mask()
        self.update_image_callback()

    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()
        if not self.probs_history:
            self.reset_init_mask()
        self._update_postprocessed_mask()
        self.update_image_callback()

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self._update_postprocessed_mask()
        self.update_image_callback()

    def finish_object(self):
        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()

    def reset_last_object(self, update_image=True):
        self.states = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self._update_postprocessed_mask()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0
        self._update_postprocessed_mask()

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        if self.probs_history:
            if self.enable_postprocessing and self._postprocessed_mask is not None:
                mask_to_apply = self._enforce_clicks_on_bool_mask(self._postprocessed_mask.copy())
            else:
                mask_to_apply = self._enforce_clicks_on_bool_mask(self.current_object_prob > self.prob_thresh)
            result_mask[mask_to_apply] = self.object_count + 1
            # 标签级别保障点击约束
            result_mask = self._enforce_clicks_on_label_mask(result_mask, fg_value=self.object_count + 1)
        return result_mask

    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        # 使用后处理掩码或原始掩码
        if self.enable_postprocessing and self._postprocessed_mask is not None:
            results_mask_for_vis = self._result_mask.copy()
            enforced = self._enforce_clicks_on_bool_mask(self._postprocessed_mask.copy())
            results_mask_for_vis[enforced] = self.object_count + 1
            results_mask_for_vis = self._enforce_clicks_on_label_mask(results_mask_for_vis,
                                                                      fg_value=self.object_count + 1)
        else:
            results_mask_for_vis = self.result_mask

        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)

        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)

        # 将新增像素的淡红色高亮放在所有掩码叠加之后，避免被后续绘制覆盖
        if self.enable_postprocessing and self._extended_pixels is not None and np.any(self._extended_pixels):
            red_overlay = np.zeros_like(vis)
            red_overlay[self._extended_pixels] = [255, 128, 128]  # 淡红色
            vis = cv2.addWeighted(vis, 1.0, red_overlay, 0.5, 0)
            print("后处理效果已应用，新增像素用淡红色标记")

        return vis

