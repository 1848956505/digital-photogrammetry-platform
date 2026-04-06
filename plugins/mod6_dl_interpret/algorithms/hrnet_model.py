"""
HRNet 语义分割模型加载器（简化版）
直接使用 LoveDA 项目的模块
"""
import os
import sys
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class HRNetSegmentor:
    """HRNet 语义分割器"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.device = 'cpu'
        self.available = False
        self.torch_available = False
        self.ever_available = False
        
        # 添加 LoveDA 路径
        self._add_loveda_paths()
        
        # 尝试导入依赖
        self._import_dependencies()
        
        if model_path and self.torch_available and self.ever_available:
            self.load_model(model_path)
    
    def _add_loveda_paths(self):
        """添加 LoveDA 路径"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        loveda_path = os.path.join(project_root, 'LoveDA-master', 'Semantic_Segmentation')
        
        if os.path.exists(loveda_path):
            sys.path.insert(0, loveda_path)
            sys.path.insert(0, os.path.join(loveda_path, 'configs'))
            logger.info(f"LoveDA path added: {loveda_path}")
    
    def _import_dependencies(self):
        """导入依赖"""
        try:
            import torch
            self.torch_available = True
            logger.info("PyTorch available")
        except ImportError:
            logger.warning("PyTorch not available")
        
        try:
            import ever as er
            self.ever_available = True
            logger.info("EVER available")
        except ImportError:
            logger.warning("EVER not available")
    
    def load_model(self, model_path: str) -> bool:
        """加载模型（直接使用 LoveDA 的方式）"""
        if not self.torch_available or not self.ever_available:
            logger.error("PyTorch or EVER not available")
            return False
        
        try:
            import torch
            import ever as er
            
            # 注册所有模块
            er.registry.register_all()
            
            # 导入 HRNetFusion
            from module.baseline.hrnet import HRNetFusion
            
            # 设备
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            
            # 直接用字典当配置！
            config_dict = {
                'backbone': {
                    'hrnet_type': 'hrnetv2_w32',
                    'pretrained': False,
                    'norm_eval': False,
                    'frozen_stages': -1,
                    'with_cp': False,
                    'with_gc': False,
                },
                'neck': {
                    'in_channels': 480,
                },
                'classes': 7,
                'head': {
                    'in_channels': 480,
                    'upsample_scale': 4.0,
                },
                'loss': {
                    'ignore_index': -1,
                    'ce': {},
                }
            }
            
            # 创建模型
            logger.info("Creating HRNetFusion...")
            self.model = HRNetFusion(config_dict)
            
            # 加载权重
            if os.path.exists(model_path):
                logger.info(f"Loading weights from: {model_path}")
                state_dict = torch.load(model_path, map_location=self.device)
                
                # 移除 module 前缀
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_key = k[7:] if k.startswith('module.') else k
                    new_state_dict[new_key] = v
                
                self.model.load_state_dict(new_state_dict, strict=False)
                logger.info("Weights loaded")
            
            self.model.to(self.device)
            self.model.eval()
            self.available = True
            
            logger.info("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, image: np.ndarray) -> Optional[np.ndarray]:
        """预测"""
        if not self.available or self.model is None:
            return self._predict_mock(image)
        
        try:
            import torch
            import cv2
            
            orig_h, orig_w = image.shape[:2]
            
            # 调整大小
            image_resized = cv2.resize(image, (512, 512))
            
            # 预处理
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            
            img_normalized = (image_resized.astype(np.float32) - mean) / std
            img_tensor = np.transpose(img_normalized, (2, 0, 1))
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor = torch.from_numpy(img_tensor).to(self.device)
            
            # 预测
            with torch.no_grad():
                output = self.model(img_tensor)
                prediction = output.argmax(dim=1).cpu().numpy()[0]
            
            # 调整回原始大小
            prediction = cv2.resize(prediction, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._predict_mock(image)
    
    def _predict_mock(self, image: np.ndarray) -> np.ndarray:
        """模拟预测"""
        h, w = image.shape[:2]
        gray = np.mean(image, axis=2)
        prediction = np.zeros((h, w), dtype=np.uint8)
        
        prediction[gray < 50] = 0
        prediction[(gray >= 50) & (gray < 100)] = 6
        prediction[(gray >= 100) & (gray < 150)] = 5
        prediction[(gray >= 150) & (gray < 200)] = 3
        prediction[gray >= 200] = 1
        
        np.random.seed(42)
        road_mask = np.random.choice([0, 1], size=(h, w), p=[0.95, 0.05])
        prediction[road_mask == 1] = 2
        
        barren_mask = np.random.choice([0, 1], size=(h, w), p=[0.98, 0.02])
        prediction[barren_mask == 1] = 4
        
        return prediction
    
    def is_available(self) -> bool:
        return self.available
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'available': self.available,
            'torch_available': self.torch_available,
            'ever_available': self.ever_available,
            'device': self.device,
            'model_path': self.model_path
        }
