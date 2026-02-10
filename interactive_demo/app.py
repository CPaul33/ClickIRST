import tkinter as tk
from tkinter import messagebox, filedialog, ttk

import os
import glob
import cv2
import numpy as np

from PIL import Image, ImageTk

from interactive_demo.canvas import CanvasImage
from interactive_demo.controller import InteractiveController
from interactive_demo.wrappers import BoundedNumericalEntry, FocusHorizontalScale, FocusCheckButton, \
    FocusButton, FocusLabelFrame


class InteractiveDemoApp(ttk.Frame):
    def __init__(self, master, args, model):
        super().__init__(master)
        self.master = master
        master.title("ClickIRST-equipped Method Demo")
        master.withdraw()
        master.update_idletasks()
        x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
        y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
        master.geometry("+%d+%d" % (x, y))
        self.pack(fill="both", expand=True)

        self.brs_modes = ['NoBRS', 'NoBRS-SegNext','RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']
        self.limit_longest_size = args.limit_longest_size

        self.controller = InteractiveController(model, args.device,
                                                predictor_params={'brs_mode': 'NoBRS'},
                                                update_image_callback=self._update_image)

        self._init_state()
        self._add_menu()
        # Create a resizable horizontal split: left is image, right is controls
        self.main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill="both", expand=True)
        self._add_canvas()
        self._add_buttons()

        # 初始化后处理状态
        self.controller.set_postprocessing_enabled(self.state['postprocessing']['enabled'].get())
        self.controller.set_edge_threshold(self.state['postprocessing']['edge_threshold'].get())

        master.bind('<space>', lambda event: self.controller.finish_object())
        master.bind('a', lambda event: self.controller.partially_finish_object())
        master.bind('q', self._prev_image)
        master.bind('e', self._next_image)
        master.bind('Q', self._prev_image)
        master.bind('E', self._next_image)
        master.bind('r', self._save_mask_hotkey)
        master.bind('R', self._save_mask_hotkey)
        master.bind('<Escape>', lambda event: self._reset_last_object())
        master.bind('<Tab>', self._toggle_preview_reset)

        self.state['zoomin_params']['skip_clicks'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['target_size'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['expansion_ratio'].trace(mode='w', callback=self._reset_predictor)
        self.state['predictor_params']['net_clicks_limit'].trace(mode='w', callback=self._change_brs_mode)
        self.state['lbfgs_max_iters'].trace(mode='w', callback=self._change_brs_mode)
        self._change_brs_mode()

        # runtime state for mini map
        self._current_vis = None
        self._mini_map_tk = None
        self._last_hover_rc = None
        # runtime toggle for preview reset (raw image view)
        self._preview_reset = False
        # image navigation state
        self._image_list = []
        self._image_index = -1

    def _init_state(self):
        self.state = {
            'zoomin_params': {
                'use_zoom_in': tk.BooleanVar(value=True),
                'fixed_crop': tk.BooleanVar(value=True),
                'skip_clicks': tk.IntVar(value=-1),
                'target_size': tk.IntVar(value=448), #tk.IntVar(value=min(400, self.limit_longest_size)),
                'expansion_ratio': tk.DoubleVar(value=1.4)
            },

            'predictor_params': {
                'net_clicks_limit': tk.IntVar(value=8)
            },
            'brs_mode': tk.StringVar(value='NoBRS'),
            'prob_thresh': tk.DoubleVar(value=0.5),
            'lbfgs_max_iters': tk.IntVar(value=20),

            'alpha_blend': tk.DoubleVar(value=0.5),
            'click_radius': tk.IntVar(value=0),

            # 添加后处理功能的状态变量
            'postprocessing': {
                'enabled': tk.BooleanVar(value=True),
                'edge_threshold': tk.DoubleVar(value=0.5)
            },
            'auto_load_mask': tk.BooleanVar(value=False)
        }

    def _add_menu(self):
        self.menubar = FocusLabelFrame(self, bd=1)
        self.menubar.pack(side=tk.TOP, fill='x')

        button = FocusButton(self.menubar, text='Load image', command=self._load_image_callback)
        button.pack(side=tk.LEFT)
        self.save_mask_btn = FocusButton(self.menubar, text='Save mask', command=self._save_mask_callback)
        self.save_mask_btn.pack(side=tk.LEFT)
        self.save_mask_btn.configure(state=tk.DISABLED)

        self.load_mask_btn = FocusButton(self.menubar, text='Load mask', command=self._load_mask_callback)
        self.load_mask_btn.pack(side=tk.LEFT)
        self.load_mask_btn.configure(state=tk.DISABLED)

        self.auto_load_checkbox = FocusCheckButton(self.menubar, text='Auto mask',
                                                   variable=self.state['auto_load_mask'],
                                                   command=self._try_auto_load_mask)
        self.auto_load_checkbox.pack(side=tk.LEFT, padx=5)

        button = FocusButton(self.menubar, text='About', command=self._about_callback)
        button.pack(side=tk.LEFT)
        button = FocusButton(self.menubar, text='Exit', command=self.master.quit)
        button.pack(side=tk.LEFT)

        # Current image filename display (right side of the menubar)
        self.current_image_label = tk.Label(self.menubar, text="Image: -", fg="black")
        self.current_image_label.pack(side=tk.LEFT, padx=10)

    def _add_canvas(self):
        # Attach image canvas to the left pane for horizontal resizing
        self.canvas_frame = FocusLabelFrame(self.main_paned, text="Image")
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, cursor="hand1", width=400, height=400)
        self.canvas.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)
        # bind mouse motion to update mini map
        self.canvas.bind('<Motion>', self._on_mouse_move)

        self.image_on_canvas = None
        # Add canvas frame to paned window so the sash can resize it
        self.main_paned.add(self.canvas_frame, weight=3)

    def _add_mini_map(self):
        # Mini Map panel moved into Controls bar top pane of a vertical splitter
        if not hasattr(self, 'controls_paned'):
            return
        self.mini_map_frame = FocusLabelFrame(self.controls_paned, text="Mini Map")
        # Add mini map frame as the top pane to ensure space for bottom controls
        self.controls_paned.add(self.mini_map_frame, weight=1)
        # Resizable canvas for mini map preview
        self.mini_map_canvas = tk.Canvas(self.mini_map_frame, highlightthickness=1, bd=1)
        self.mini_map_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # Re-render mini map when the canvas is resized
        self.mini_map_canvas.bind('<Configure>', self._on_mini_map_resize)

    def _add_buttons(self):
        # Attach controls to the right pane for horizontal resizing
        self.control_frame = FocusLabelFrame(self.main_paned, text="Controls")
        self.main_paned.add(self.control_frame, weight=1)
        # Inside Controls, use a vertical splitter: top is Mini Map, bottom is the rest
        self.controls_paned = ttk.PanedWindow(self.control_frame, orient=tk.VERTICAL)
        self.controls_paned.pack(fill=tk.BOTH, expand=True)
        # Add Mini Map to the top pane
        self._add_mini_map()
        # Bottom pane holds all other controls
        self.controls_content = ttk.Frame(self.controls_paned)
        self.controls_paned.add(self.controls_content, weight=3)
        master = self.controls_content

        self.clicks_options_frame = FocusLabelFrame(master, text="Clicks management")
        self.clicks_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.finish_object_button = \
            FocusButton(self.clicks_options_frame, text='Finish\nobject', bg='#b6d7a8', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self.controller.finish_object)
        self.finish_object_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.undo_click_button = \
            FocusButton(self.clicks_options_frame, text='Undo click', bg='#ffe599', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self.controller.undo_click)
        self.undo_click_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.reset_clicks_button = \
            FocusButton(self.clicks_options_frame, text='Reset clicks', bg='#ea9999', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self._reset_last_object)
        self.reset_clicks_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        self.zoomin_options_frame = FocusLabelFrame(master, text="ZoomIn options")
        self.zoomin_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusCheckButton(self.zoomin_options_frame, text='Use ZoomIn', command=self._reset_predictor,
                         variable=self.state['zoomin_params']['use_zoom_in']).grid(row=0, column=0, padx=10)
        FocusCheckButton(self.zoomin_options_frame, text='Fixed crop', command=self._reset_predictor,
                         variable=self.state['zoomin_params']['fixed_crop']).grid(row=1, column=0, padx=10)
        tk.Label(self.zoomin_options_frame, text="Skip clicks").grid(row=0, column=1, pady=1, sticky='e')
        tk.Label(self.zoomin_options_frame, text="Target size").grid(row=1, column=1, pady=1, sticky='e')
        tk.Label(self.zoomin_options_frame, text="Expand ratio").grid(row=2, column=1, pady=1, sticky='e')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['skip_clicks'],
                              min_value=-1, max_value=None, vartype=int,
                              name='zoom_in_skip_clicks').grid(row=0, column=2, padx=10, pady=1, sticky='w')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['target_size'],
                              min_value=100, max_value=self.limit_longest_size, vartype=int,
                              name='zoom_in_target_size').grid(row=1, column=2, padx=10, pady=1, sticky='w')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['expansion_ratio'],
                              min_value=1.0, max_value=2.0, vartype=float,
                              name='zoom_in_expansion_ratio').grid(row=2, column=2, padx=10, pady=1, sticky='w')
        self.zoomin_options_frame.columnconfigure((0, 1, 2), weight=1)

        # BRS options removed
        self.prob_thresh_frame = FocusLabelFrame(master, text="Predictions threshold")
        self.prob_thresh_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.prob_thresh_frame, from_=0.0, to=1.0, command=self._update_prob_thresh,
                             variable=self.state['prob_thresh']).pack(padx=10)

        self.alpha_blend_frame = FocusLabelFrame(master, text="Alpha blending coefficient")
        self.alpha_blend_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.alpha_blend_frame, from_=0.0, to=1.0, command=self._update_blend_alpha,
                             variable=self.state['alpha_blend']).pack(padx=10, anchor=tk.CENTER)

        self.click_radius_frame = FocusLabelFrame(master, text="Visualisation click radius")
        self.click_radius_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.click_radius_frame, from_=0, to=7, resolution=1, command=self._update_click_radius,
                             variable=self.state['click_radius']).pack(padx=10, anchor=tk.CENTER)

        # 添加后处理功能控件
        self.postprocessing_frame = FocusLabelFrame(master, text="Post-processing")
        self.postprocessing_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)

        # 后处理启用开关
        FocusCheckButton(self.postprocessing_frame, text='Enable edge extension',
                         command=self._update_postprocessing,
                         variable=self.state['postprocessing']['enabled']).pack(padx=10, pady=2)

        # 边缘阈值滑块
        threshold_frame = tk.Frame(self.postprocessing_frame)
        threshold_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(threshold_frame, text="Edge threshold:").pack(side=tk.LEFT)
        FocusHorizontalScale(threshold_frame, from_=0.1, to=1.0, resolution=0.1,
                             command=self._update_edge_threshold,
                             variable=self.state['postprocessing']['edge_threshold']).pack(side=tk.RIGHT, padx=5)

        # Esc hint: matches Tab hint styling and placement
        self.esc_hint_label = tk.Label(master,
                                       text="Tip: Press <Esc> to reset clicks.",
                                       fg="red")
        self.esc_hint_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)

        self.r_hint_label = tk.Label(master,
                                      text="Press <R> to save current prediction mask.",
                                      fg="red")
        self.r_hint_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)

        self.qe_hint_label = tk.Label(master,
                                      text="Press <Q>/<E> to switch previous/next image.",
                                      fg="red")
        self.qe_hint_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)

        self.a_hint_label = tk.Label(master,
                                     text="Press <A> to partially finish current object.",
                                     fg="red")
        self.a_hint_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)

        self.space_hint_label = tk.Label(master,
                                         text="Press <Space> to finish current object.",
                                         fg="red")
        self.space_hint_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)


    def _load_image_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*"),
            ], title="Chose an image")

            if len(filename) > 0:
                image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                self.controller.set_image(image)
                self.save_mask_btn.configure(state=tk.NORMAL)
                self.load_mask_btn.configure(state=tk.NORMAL)
                # build image list from the directory of selected file
                self._build_image_list(filename)
                # set current index to the selected file
                try:
                    norm_files = [os.path.normcase(os.path.normpath(p)) for p in self._image_list]
                    sel = os.path.normcase(os.path.normpath(filename))
                    self._image_index = norm_files.index(sel)
                except ValueError:
                    self._image_index = max(0, self._image_index)
                # Update filename label after loading image and setting index
                self._update_current_image_label()
                self._try_auto_load_mask()

    def _try_auto_load_mask(self):
        """
        Attempt to automatically load a mask for the current image if the checkbox is checked.
        The mask is expected to be in a 'masks' folder in the parent directory of the image's directory.
        """
        if not self.state['auto_load_mask'].get():
            return

        # Check if the model supports loading masks
        if not self.controller.net.with_prev_mask:
            print("Model does not support external masks. Auto-load skipped.")
            return

        if not hasattr(self, '_image_list') or not self._image_list or self._image_index is None:
            return

        try:
            current_path = self._image_list[self._image_index]
        except Exception:
            return

        # Construct mask path: ../masks/<basename>
        img_dir = os.path.dirname(current_path)
        parent_dir = os.path.dirname(img_dir)
        masks_dir = os.path.join(parent_dir, 'masks')
        
        if not os.path.exists(masks_dir):
            return

        basename = os.path.basename(current_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        # Candidate filenames: exact match, or png/bmp with same name
        candidates = [
            basename,
            f"{name_no_ext}.png",
            f"{name_no_ext}.bmp"
        ]

        mask_path = None
        for cand in candidates:
            p = os.path.join(masks_dir, cand)
            if os.path.exists(p):
                mask_path = p
                break
        
        if mask_path:
            try:
                # Load mask using same logic as _load_mask_callback
                # Read as grayscale
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    # Check size
                    if self.controller.image is not None:
                        h, w = self.controller.image.shape[:2]
                        if mask_img.shape[:2] != (h, w):
                            print(f"Skipping auto-load for {mask_path}: size mismatch {mask_img.shape[:2]} vs {(h, w)}")
                            return

                    # Binarize
                    mask = mask_img > 127
                    self.controller.set_mask(mask)
                    self._update_image()
                    print(f"Auto-loaded mask: {mask_path}")
            except Exception as e:
                print(f"Failed to auto-load mask {mask_path}: {e}")

    def _save_mask_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            mask = self.controller.result_mask
            if mask is None:
                return

            filename = filedialog.asksaveasfilename(parent=self.master, initialfile='mask.png', filetypes=[
                ("PNG image", "*.png"),
                ("BMP image", "*.bmp"),
                ("All files", "*.*"),
            ], title="Save the current mask as...")

            if len(filename) > 0:
                # Save as strict binary mask: union of all objects -> {0,255}
                mask_bin = (mask > 0).astype(np.uint8) * 255
                cv2.imwrite(filename, mask_bin)

    def _save_mask_hotkey(self, event=None):
        """Save current prediction mask into <current_image_dir>/predictions/<current_filename> on 'R' key."""
        # Ensure we have a mask to save
        mask = self.controller.result_mask
        if mask is None:
            return

        # Determine current image path from list/index
        if not hasattr(self, '_image_list') or not self._image_list or self._image_index is None:
            return
        try:
            current_path = self._image_list[self._image_index]
        except Exception:
            return

        # Build predictions directory under the parent of the current image's directory
        img_dir = os.path.dirname(current_path)
        parent_dir = os.path.dirname(img_dir)
        pred_dir = os.path.join(parent_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)

        # Use the same filename (including extension) as the current image
        save_path = os.path.join(pred_dir, os.path.basename(current_path))

        # Save as strict binary mask: union of all objects -> {0,255}
        m = (mask > 0).astype(np.uint8) * 255

        # Write mask to disk and notify user
        success = False
        try:
            success = cv2.imwrite(save_path, m)
        except Exception:
            success = False
        if success:
            self._show_toast(f"Mask saved to:\n{save_path}", bg="#28a745", fg="white", duration=2000)
        else:
            self._show_toast(f"Save failed:\n{save_path}", bg="#dc3545", fg="white", duration=2500)

    def _show_toast(self, text, bg="#333333", fg="white", duration=2000, y_offset=50):
        """Show a temporary toast message near the top of the screen that auto-dismisses."""
        toast = tk.Toplevel(self.master)
        toast.overrideredirect(True)
        try:
            toast.attributes('-topmost', True)
        except Exception:
            pass

        label = tk.Label(toast, text=text, bg=bg, fg=fg, padx=12, pady=8, justify=tk.LEFT)
        label.pack()

        toast.update_idletasks()
        screen_w = self.master.winfo_screenwidth()
        w = toast.winfo_reqwidth()
        h = toast.winfo_reqheight()
        x = int((screen_w - w) / 2)
        y = y_offset
        toast.geometry(f"{w}x{h}+{x}+{y}")

        toast.after(duration, toast.destroy)

    def _load_mask_callback(self):
        if not self.controller.net.with_prev_mask:
            messagebox.showwarning("Warning", "The current model doesn't support loading external masks. "
                                              "Please use ITER-M models for that purpose.")
            return

        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Binary mask (png, bmp)", "*.png *.bmp"),
                ("All files", "*.*"),
            ], title="Chose an image")

            if len(filename) > 0:
                mask = cv2.imread(filename)[:, :, 0] > 127
                self.controller.set_mask(mask)
                self._update_image()

    def _about_callback(self):
        self.menubar.focus_set()

        text = [
            "Developed by:",
            "K.Sofiiuk and I. Petrov",
            "The MIT License, 2021"
        ]

        messagebox.showinfo("About Demo", '\n'.join(text))

    def _reset_last_object(self):
        self.state['alpha_blend'].set(0.5)
        self.state['prob_thresh'].set(0.5)
        self.controller.reset_last_object()

    def _update_prob_thresh(self, value):
        if self.controller.is_incomplete_mask:
            # 使用控制器方法以便在阈值改变时同步触发后处理刷新
            self.controller.set_prob_thresh(self.state['prob_thresh'].get())
            self._update_image()

    def _update_blend_alpha(self, value):
        self._update_image()

    def _update_click_radius(self, *args):
        if self.image_on_canvas is None:
            return

        self._update_image()

    def _change_brs_mode(self, *args):
        self._reset_predictor()

    def _reset_predictor(self, *args, **kwargs):
        brs_mode = self.state['brs_mode'].get()
        prob_thresh = self.state['prob_thresh'].get()
        net_clicks_limit = None if brs_mode == 'NoBRS' else self.state['predictor_params']['net_clicks_limit'].get()

        if self.state['zoomin_params']['use_zoom_in'].get():
            zoomin_params = {
                'skip_clicks': self.state['zoomin_params']['skip_clicks'].get(),
                'target_size': self.state['zoomin_params']['target_size'].get(),
                'expansion_ratio': self.state['zoomin_params']['expansion_ratio'].get()
            }
            if self.state['zoomin_params']['fixed_crop'].get():
                zoomin_params['target_size'] = (zoomin_params['target_size'], zoomin_params['target_size'])
        else:
            zoomin_params = None

        predictor_params = {
            'brs_mode': brs_mode,
            'prob_thresh': prob_thresh,
            'zoom_in_params': zoomin_params,
            'predictor_params': {
                'net_clicks_limit': net_clicks_limit,
                'max_size': self.limit_longest_size
            },
            'brs_opt_func_params': {'min_iou_diff': 1e-3},
            'lbfgs_params': {'maxfun': self.state['lbfgs_max_iters'].get()}
        }
        self.controller.reset_predictor(predictor_params)

    def _click_callback(self, is_positive, x, y):
        self.canvas.focus_set()

        if self.image_on_canvas is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        if self._check_entry(self):
            self.controller.add_click(x, y, is_positive)

    def _update_image(self, reset_canvas=False):
        if self._preview_reset and getattr(self.controller, 'image', None) is not None:
            image = self.controller.image
        else:
            image = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'].get(),
                                                  click_radius=self.state['click_radius'].get())
        # keep latest visualization for mini map extraction
        self._current_vis = image
        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)
            self.image_on_canvas.register_click_callback(self._click_callback)

        self._set_click_dependent_widgets_state()
        if image is not None:
            self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)

    def _toggle_preview_reset(self, event=None):
        # Toggle between raw image (no annotations) and current prediction visualization
        if getattr(self.controller, 'image', None) is None:
            return
        self._preview_reset = not self._preview_reset
        # refresh image without changing canvas geometry
        self._update_image(reset_canvas=False)

    def _build_image_list(self, start_path):
        # Collect supported images in the same directory as start_path
        dirpath = os.path.dirname(start_path)
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif")
        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(dirpath, pat)))
        # sort for stable navigation
        files = sorted(files, key=lambda p: os.path.basename(p).lower())
        self._image_list = files
        # default to first item if not set
        if not self._image_list:
            self._image_index = -1
        elif self._image_index < 0:
            self._image_index = 0

    def _switch_image(self, delta):
        # Navigate to previous/next image in the directory list
        if not self._image_list:
            messagebox.showwarning("Warning", "Please load an image first to initialize the directory list.")
            return
        n = len(self._image_list)
        if n == 0:
            return
        self._image_index = (self._image_index + delta) % n
        path = self._image_list[self._image_index]
        # load image by path and reset controller state via set_image
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        self.controller.set_image(image)
        self.save_mask_btn.configure(state=tk.NORMAL)
        self.load_mask_btn.configure(state=tk.NORMAL)
        # Update filename label after switching image
        self._update_current_image_label()
        self._try_auto_load_mask()

    def _prev_image(self, event=None):
        self._switch_image(-1)

    def _next_image(self, event=None):
        self._switch_image(1)

    def _update_current_image_label(self):
        """Update the menubar label to show the current image filename."""
        try:
            if hasattr(self, '_image_list') and self._image_list and isinstance(self._image_index, int) \
                    and 0 <= self._image_index < len(self._image_list):
                current_path = self._image_list[self._image_index]
                base = os.path.basename(current_path)
                self.current_image_label.configure(text=f"Image: {base}")
            else:
                self.current_image_label.configure(text="Image: -")
        except Exception:
            self.current_image_label.configure(text="Image: -")

    def _on_mouse_move(self, event):
        # Update mini map with a local crop around the cursor, if inside image
        if self.image_on_canvas is None or self._current_vis is None:
            return

        # use CanvasImage internal mapping to convert canvas coords to image coords
        try:
            coords = self.image_on_canvas._get_click_coordinates(event)
        except Exception:
            coords = None

        if coords is None:
            return

        # CanvasImage returns (x, y) = (col, row). Convert to (row, col).
        col, row = coords[0], coords[1]
        self._last_hover_rc = (row, col)
        self._render_mini_map(row, col)

    def _on_mini_map_resize(self, event):
        # When the mini map canvas resizes, refresh its content based on last hover coords
        if self._last_hover_rc is None:
            return
        row, col = self._last_hover_rc
        self._render_mini_map(row, col)

    def _render_mini_map(self, row, col):
        # crop a square region around (x, y) from the latest visualization
        img = self._current_vis
        if img is None:
            return

        h, w = img.shape[:2]
        radius = 10  # increase magnification by reducing crop size
        r1 = max(0, row - radius)
        c1 = max(0, col - radius)
        r2 = min(h, row + radius)
        c2 = min(w, col + radius)

        patch = img[r1:r2, c1:c2]
        if patch.size == 0:
            return

        # resize patch to current mini map canvas size (proportional scaling)
        pil_patch = Image.fromarray(patch)
        mw = max(1, int(self.mini_map_canvas.winfo_width()))
        mh = max(1, int(self.mini_map_canvas.winfo_height()))
        pil_patch = pil_patch.resize((mw, mh), Image.Resampling.NEAREST)
        self._mini_map_tk = ImageTk.PhotoImage(pil_patch)
        self.mini_map_canvas.delete('all')
        self.mini_map_canvas.create_image(0, 0, anchor='nw', image=self._mini_map_tk)

        # draw crosshair at the mouse-relative position inside the mini map
        ph = max(1, r2 - r1)
        pw = max(1, c2 - c1)
        cx = int((col - c1) * (mw / pw))
        cy = int((row - r1) * (mh / ph))
        size = max(4, min(mw, mh) // 20)
        self.mini_map_canvas.create_line(cx - size, cy, cx + size, cy, fill='red', width=1)
        self.mini_map_canvas.create_line(cx, cy - size, cx, cy + size, fill='red', width=1)
        self.mini_map_canvas.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, outline='red', width=1)

    def _set_click_dependent_widgets_state(self):
        after_1st_click_state = tk.NORMAL if self.controller.is_incomplete_mask else tk.DISABLED
        before_1st_click_state = tk.DISABLED if self.controller.is_incomplete_mask else tk.NORMAL

        self.finish_object_button.configure(state=after_1st_click_state)
        self.undo_click_button.configure(state=after_1st_click_state)
        self.reset_clicks_button.configure(state=after_1st_click_state)
        self.zoomin_options_frame.set_frame_state(before_1st_click_state)

    def _check_entry(self, widget):
        all_checked = True
        if widget.winfo_children is not None:
            for w in widget.winfo_children():
                all_checked = all_checked and self._check_entry(w)

        if getattr(widget, "_check_bounds", None) is not None:
            all_checked = all_checked and widget._check_bounds(widget.get(), '-1')

        return all_checked

    def _update_postprocessing(self, *args):
        # 更新控制器的后处理状态
        self.controller.set_postprocessing_enabled(self.state['postprocessing']['enabled'].get())
        self.controller.set_edge_threshold(self.state['postprocessing']['edge_threshold'].get())
        self._update_image()

    def _update_edge_threshold(self, value):
        # 更新边缘阈值
        self.controller.set_edge_threshold(self.state['postprocessing']['edge_threshold'].get())
        self._update_image()
