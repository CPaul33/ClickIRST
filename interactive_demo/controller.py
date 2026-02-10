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
        self._postprocessed_mask = None  # Store mask after post-processing
        self._extended_pixels = None  # Store newly added foreground pixels
        self.enable_postprocessing = False  # Switch for post-processing function
        self.edge_threshold = 0.5  # Threshold for outer edge gray value

        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()

    def _enforce_clicks_on_bool_mask(self, mask):
        """Enforce click constraints on boolean mask:
        - Positive clicks set to True; Negative clicks set to False
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
        """Enforce click constraints on label mask:
        - Positive clicks set to foreground label; Negative clicks set to background label
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
            # Changes in probability threshold affect original mask and average gray statistics, need to re-calculate post-processing results
            self._update_postprocessed_mask()
            self.update_image_callback()

    def _update_postprocessed_mask(self):
        """Update post-processed mask"""
        if not self.enable_postprocessing or self.image is None:
            self._postprocessed_mask = None
            self._extended_pixels = None
            return

        current_prob = self.current_object_prob
        if current_prob is None:
            self._postprocessed_mask = None
            self._extended_pixels = None
            return

        # Get original mask, and enforce strong click constraints before post-processing
        original_mask = current_prob > self.prob_thresh
        original_mask = self._enforce_clicks_on_bool_mask(original_mask)

        # Calculate average gray value of foreground area
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        foreground_pixels = gray_image[original_mask]
        if len(foreground_pixels) == 0:
            self._postprocessed_mask = original_mask
            self._extended_pixels = None
            return

        avg_gray_value = np.mean(foreground_pixels)

        # Calculate "outer edge": Dilate foreground once, take difference between dilated result and original mask to get a ring of background pixels adjacent to foreground
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(original_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        outer_border = np.logical_and(dilated, np.logical_not(original_mask))

        # Initialize extended mask and new pixel markers
        extended_mask = original_mask.copy()
        extended_pixels = np.zeros_like(original_mask, dtype=bool)

        # Vectorized check if edge pixels need to be included in foreground
        if avg_gray_value > 0:
            by, bx = np.where(outer_border)
            if by.size > 0:
                border_gray = gray_image[by, bx].astype(np.float32)
                ratio = border_gray / float(avg_gray_value)
                take = ratio > float(self.edge_threshold)
                if np.any(take):
                    extended_mask[by[take], bx[take]] = True
                    extended_pixels[by[take], bx[take]] = True
                    print(f"Post-processing algorithm extended {int(np.count_nonzero(take))} pixels")
                else:
                    print("Post-processing algorithm extended 0 pixels")
            else:
                print("Post-processing algorithm extended 0 pixels")
        else:
            # Extreme case: Foreground average gray is 0, cannot compare, keep as is
            print("Foreground average gray is 0, no edge extension performed")

        # Enforce strong click constraints again after post-processing to prevent extension from covering negative points or clearing positive points
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
            # Ensure click constraints at label level
            result_mask = self._enforce_clicks_on_label_mask(result_mask, fg_value=self.object_count + 1)
        return result_mask

    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        # Use post-processed mask or original mask
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

        # Place light red highlight of new pixels after all mask overlays to avoid being covered by subsequent drawing
        if self.enable_postprocessing and self._extended_pixels is not None and np.any(self._extended_pixels):
            red_overlay = np.zeros_like(vis)
            red_overlay[self._extended_pixels] = [255, 128, 128]  # Light red
            vis = cv2.addWeighted(vis, 1.0, red_overlay, 0.5, 0)
            print("Post-processing effect applied, new pixels marked in light red")

        return vis
