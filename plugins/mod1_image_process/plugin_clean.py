from __future__ import annotations
import os, cv2, numpy as np
from typing import Dict, Any, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox, QFormLayout, QComboBox, QSpinBox, QFileDialog, QMessageBox, QHBoxLayout, QSlider
from PySide6.QtCore import Qt, Signal, QThread
from core.base_interface import IPlugin
from core.event_bus import EventTopics, get_event_bus
from core.log_manager import log_manager
from plugins.mod1_image_process.algorithms.grayscale import GrayscaleProcessor, ResampleProcessor
from plugins.mod1_image_process.algorithms.filters import FilterProcessor
from plugins.mod1_image_process.algorithms.features import FeatureDetector
from plugins.mod1_image_process.algorithms.matcher import CompareMatcher, FeatureMatcher

def _to_bgr(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return arr.copy()

class ProcessingThread(QThread):
    progress = Signal(int); finished = Signal(object); error = Signal(str)
    def __init__(self, func, *args, **kwargs): super().__init__(); self.func=func; self.args=args; self.kwargs=kwargs
    def run(self):
        try: self.progress.emit(10); r=self.func(*self.args, **self.kwargs); self.progress.emit(100); self.finished.emit(r)
        except Exception as e: self.error.emit(str(e))

class ImageProcessPlugin(IPlugin):
    def __init__(self, workspace):
        super().__init__(workspace); self.current_image=None; self.processed_image=None; self.image1_path=None; self.image2_path=None; self._event_bus=get_event_bus(); self._panel=None
    def plugin_info(self)->Dict[str,Any]: return {'name':'图像处理模块','group':'模块','version':'1.0.0','description':'具备真实可视化能力的图像处理模块。'}
    def get_ui_panel(self)->QWidget:
        if self._panel is not None: return self._panel
        w = QWidget(); l = QVBoxLayout(w)
        l.addWidget(QLabel('图像处理模块'))
        g1 = QGroupBox('灰度与增强'); f1 = QFormLayout(); self.gray_method=QComboBox(); self.gray_method.addItems(['线性变换','对数变换','指数变换','直方图均衡','CLAHE']); f1.addRow('方法', self.gray_method); self.gray_alpha=QSlider(Qt.Orientation.Horizontal); self.gray_alpha.setRange(50,200); self.gray_alpha.setValue(100); f1.addRow('亮度系数', self.gray_alpha); self.gray_beta=QSlider(Qt.Orientation.Horizontal); self.gray_beta.setRange(-100,100); self.gray_beta.setValue(0); f1.addRow('偏移量', self.gray_beta); b=QPushButton('执行灰度变换'); b.clicked.connect(self._apply_gray_transform); f1.addRow('', b); self.gray_result=QLabel('就绪'); f1.addRow('结果', self.gray_result); g1.setLayout(f1); l.addWidget(g1)
        g2=QGroupBox('滤波'); f2=QFormLayout(); self.filter_type=QComboBox(); self.filter_type.addItems(['均值滤波','中值滤波','高斯滤波','双边滤波','锐化','反锐化掩膜']); f2.addRow('类型', self.filter_type); self.filter_size=QSpinBox(); self.filter_size.setRange(3,15); self.filter_size.setValue(3); self.filter_size.setSingleStep(2); f2.addRow('核大小', self.filter_size); b=QPushButton('执行滤波'); b.clicked.connect(self._apply_filter); f2.addRow('', b); self.filter_result=QLabel('就绪'); f2.addRow('结果', self.filter_result); g2.setLayout(f2); l.addWidget(g2)
        g3=QGroupBox('特征提取'); f3=QFormLayout(); self.feature_type=QComboBox(); self.feature_type.addItems(['SIFT','Harris','Shi-Tomasi','ORB']); f3.addRow('算法', self.feature_type); self.feature_count=QSpinBox(); self.feature_count.setRange(100,5000); self.feature_count.setValue(1000); f3.addRow('数量', self.feature_count); b=QPushButton('检测特征'); b.clicked.connect(self._detect_features); f3.addRow('', b); self.feature_result=QLabel('就绪'); self.feature_result.setWordWrap(True); f3.addRow('结果', self.feature_result); g3.setLayout(f3); l.addWidget(g3)
        g4=QGroupBox('特征匹配对比'); v4=QVBoxLayout(); h=QHBoxLayout(); self.btn_select_img1=QPushButton('选择影像 1'); self.btn_select_img1.clicked.connect(lambda: self._select_image(1)); h.addWidget(self.btn_select_img1); self.btn_select_img2=QPushButton('选择影像 2'); self.btn_select_img2.clicked.connect(lambda: self._select_image(2)); h.addWidget(self.btn_select_img2); v4.addLayout(h); self.image_paths_label=QLabel('请选择两张影像'); v4.addWidget(self.image_paths_label); self.btn_compare=QPushButton('执行对比'); self.btn_compare.clicked.connect(self._run_comparison); v4.addWidget(self.btn_compare); self.match_result=QLabel('就绪'); self.match_result.setWordWrap(True); v4.addWidget(self.match_result); g4.setLayout(v4); l.addWidget(g4)
        g5=QGroupBox('重采样'); f5=QFormLayout(); self.resample_method=QComboBox(); self.resample_method.addItems(['最近邻','双线性','双三次','LANCZOS']); f5.addRow('方法', self.resample_method); self.resample_scale=QSpinBox(); self.resample_scale.setRange(10,500); self.resample_scale.setValue(100); self.resample_scale.setSuffix(' %'); f5.addRow('缩放', self.resample_scale); self.btn_resample=QPushButton('执行重采样'); self.btn_resample.clicked.connect(self._apply_resample); f5.addRow('', self.btn_resample); self.resample_result=QLabel('就绪'); f5.addRow('结果', self.resample_result); g5.setLayout(f5); l.addWidget(g5)
        self.status_label=QLabel('空闲'); l.addWidget(self.status_label); l.addStretch(1); self._panel=w; return w
    def _select_image(self, num: int):
        path,_=QFileDialog.getOpenFileName(None, f'选择影像 {num}', '', '图像文件 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)')
        if path:
            if num==1: self.image1_path=path
            else: self.image2_path=path
            n=[]
            if self.image1_path: n.append(os.path.basename(self.image1_path))
            if self.image2_path: n.append(os.path.basename(self.image2_path))
            self.image_paths_label.setText('已选择：' + ' + '.join(n) if n else '请选择两张影像')
    def _get_current_image(self)->Optional[np.ndarray]:
        images=self.workspace.get('images', {})
        if images:
            first=next(iter(images.values()))
            if isinstance(first, dict) and 'path' in first and os.path.exists(first['path']):
                return cv2.imdecode(np.fromfile(first['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
        return None
    def _update_workspace_image(self, image: np.ndarray, process_type: str='Result'):
        from datetime import datetime
        name=f'{process_type}_{datetime.now().strftime("%H%M%S")}'
        self.workspace.add_processed_image(name, image)
        path=self.workspace.get_processed_image(name)
        if path: self._event_bus.publish(EventTopics.TOPIC_IMAGE_UPDATED, {'path': path, 'name': name})
        return name
    def _publish_single_view(self, image: np.ndarray, title: str, keypoints=None, mask=None):
        self._event_bus.publish(EventTopics.TOPIC_VIEW_SINGLE_REQUEST, {'image': image, 'title': title, 'keypoints': keypoints, 'mask': mask})
    def _publish_compare_view(self, left, right, title: str, left_name: str='', right_name: str='', matches=None, sync: bool=True):
        self._event_bus.publish(EventTopics.TOPIC_VIEW_COMPARE_REQUEST, {'left': left, 'right': right, 'title': title, 'left_name': left_name, 'right_name': right_name, 'matches': matches or [], 'sync': sync})
    def _keypoints_to_points(self, keypoints): return [kp.pt for kp in keypoints or []]
    def _estimate_match_stats(self, keypoints1, keypoints2, matches):
        if len(matches) < 4: return {'inliers': 0, 'inlier_rate': 0.0, 'reproj_error': None, 'mask': None}
        src=np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1,1,2); dst=np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1,1,2); H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
        if H is None or mask is None: return {'inliers': 0, 'inlier_rate': 0.0, 'reproj_error': None, 'mask': None}
        inlier_mask=mask.ravel().astype(bool); inliers=int(inlier_mask.sum()); rate=inliers/max(len(matches),1); proj=cv2.perspectiveTransform(src,H); err=np.linalg.norm(proj.reshape(-1,2)-dst.reshape(-1,2), axis=1); err=float(err[inlier_mask].mean()) if inliers>0 else float(err.mean()); return {'inliers': inliers, 'inlier_rate': rate, 'reproj_error': err, 'mask': inlier_mask.tolist(), 'homography': H}
    def _apply_gray_transform(self):
        image=self._get_current_image();
        if image is None: return
        method=self.gray_method.currentText(); alpha=self.gray_alpha.value()/100.0; beta=self.gray_beta.value(); original=image.copy()
        if method=='线性变换': result=GrayscaleProcessor.linear_transform(image, alpha, beta)
        elif method=='对数变换': result=GrayscaleProcessor.log_transform(image)
        elif method=='指数变换': result=GrayscaleProcessor.exp_transform(image, alpha)
        elif method=='直方图均衡': result=GrayscaleProcessor.histogram_equalization(image)
        else: result=GrayscaleProcessor.clahe(image)
        self.processed_image=result; self._update_workspace_image(result, f'Gray_{method}'); self._publish_compare_view({'image': original, 'name': '原图'}, {'image': result, 'name': method}, title=f'灰度对比 - {method}', sync=False); self.gray_result.setText(f'{method} 完成'); self.status_label.setText(f'灰度处理完成：{method}')
    def _apply_filter(self):
        image=self._get_current_image();
        if image is None: return
        t=self.filter_type.currentText(); k=self.filter_size.value(); original=image.copy()
        if t=='均值滤波': result=FilterProcessor.mean_filter(image, k)
        elif t=='中值滤波': result=FilterProcessor.median_filter(image, k)
        elif t=='高斯滤波': result=FilterProcessor.gaussian_filter(image, k)
        elif t=='双边滤波': result=FilterProcessor.bilateral_filter(image)
        elif t=='锐化': result=FilterProcessor.sharpen(image)
        else: result=FilterProcessor.unsharp_mask(image)
        self.processed_image=result; self._update_workspace_image(result, f'Filter_{t}'); self._publish_compare_view({'image': original, 'name': '原图'}, {'image': result, 'name': t}, title=f'滤波对比 - {t}', sync=False); self.filter_result.setText(f'{t} 完成'); self.status_label.setText(f'滤波处理完成：{t}')
    def _detect_features(self):
        image=self._get_current_image();
        if image is None: return
        method=self.feature_type.currentText(); n=self.feature_count.value(); result=FeatureDetector.detect(image, method=method.lower().replace('-', '_'), n_features=n); desc=result.get('descriptors'); info='无' if desc is None else f'{desc.shape[0]} × {desc.shape[1]}'; self._publish_single_view(image, title=f'特征点 - {result["algorithm"]}', keypoints=self._keypoints_to_points(result.get('keypoints'))); self.feature_result.setText(f'算法：{result["algorithm"]}\n特征点数：{result["num_keypoints"]}\n描述子：{info}\n耗时：{result["time"]:.3f}s'); self.status_label.setText(f'特征提取完成：{result["algorithm"]}')
    def _build_resample_montage(self, original: np.ndarray, results: Dict[str, np.ndarray], selected_method: str) -> np.ndarray:
        labels=[('最近邻','最近邻'),('双线性','双线性'),('双三次','双三次'),('LANCZOS','LANCZOS')]
        def lab(img, text, hl=False):
            c=_to_bgr(img).copy(); b=(46,125,50) if hl else (180,180,180); t=3 if hl else 1; cv2.rectangle(c,(0,0),(c.shape[1]-1,c.shape[0]-1),b,t); cv2.rectangle(c,(0,0),(c.shape[1],28),(245,245,245),-1); cv2.putText(c,text,(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(30,30,30),2,cv2.LINE_AA); return c
        ref_h, ref_w = next(iter(results.values())).shape[:2]; original_resized=cv2.resize(_to_bgr(original),(ref_w,ref_h),interpolation=cv2.INTER_LINEAR); tiles=[lab(original_resized, 'Original')]
        for key,label in labels: tiles.append(lab(results[key], label, key==selected_method))
        return np.vstack([np.hstack([tiles[0], tiles[1]]), np.hstack([tiles[2], tiles[3]])])
    def _run_comparison(self):
        if not self.image1_path or not self.image2_path: return
        img1=cv2.imdecode(np.fromfile(self.image1_path,dtype=np.uint8),cv2.IMREAD_COLOR); img2=cv2.imdecode(np.fromfile(self.image2_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        if img1 is None or img2 is None: return
        comp=CompareMatcher().compare(img1, img2); s=comp['sift']; d=comp['dl']; ss=self._estimate_match_stats(s['keypoints1'], s['keypoints2'], s['matches']); ds=self._estimate_match_stats(d['keypoints1'], d['keypoints2'], d['matches']); sift_vis=FeatureMatcher.draw_matches_colored(img1, s['keypoints1'], img2, s['keypoints2'], s['matches'], max_matches=120, inlier_mask=ss.get('mask')); dl_vis=FeatureMatcher.draw_matches_colored(img1, d['keypoints1'], img2, d['keypoints2'], d['matches'], max_matches=120, inlier_mask=ds.get('mask')); self._publish_compare_view({'image': sift_vis, 'name': 'SIFT'}, {'image': dl_vis, 'name': 'ORB 模拟对比'}, title='SIFT 与 ORB 模拟对比', sync=False); self.match_result.setText(f'SIFT 匹配点：{comp["comparison"]["sift_matches"]}\nSIFT 内点：{ss["inliers"]}\nSIFT 内点率：{ss["inlier_rate"]:.2%}\nORB 匹配点：{comp["comparison"]["dl_matches"]}\nORB 内点：{ds["inliers"]}\nORB 内点率：{ds["inlier_rate"]:.2%}\n说明：当前 ORB 仅为模拟对比，不是 SuperGlue / LoFTR。'); self.status_label.setText('匹配对比完成')
    def _apply_resample(self):
        image=self._get_current_image();
        if image is None: return
        scale=self.resample_scale.value()/100.0; methods=['最近邻','双线性','双三次','LANCZOS']; method=methods[self.resample_method.currentIndex()]; map_method={'最近邻':'nearest','双线性':'bilinear','双三次':'bicubic','LANCZOS':'lanczos'}
        results={k: ResampleProcessor.resize(image, scale, v) for k,v in map_method.items()}; result=results[method]; self.processed_image=result; self._update_workspace_image(result, f'Resample_{int(scale*100)}%'); compare_canvas=self._build_resample_montage(image, results, method); self._publish_compare_view({'image': image, 'name': '原图'}, {'image': compare_canvas, 'name': '重采样对比'}, title=f'重采样对比 - {method}', sync=False); self.resample_result.setText(f'{method} 完成'); self.status_label.setText(f'重采样完成：{method}')
    def execute(self, *args, **kwargs): return {'success': True, 'message': 'ok'}
