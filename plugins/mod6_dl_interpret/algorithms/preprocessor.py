"""
图像预处理器
用于语义分割前的图像预处理
"""
import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """图像预处理器"""
    
    # LoveDA 数据集的均值和标准差 (RGB)
    MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    
    @staticmethod
    def read_image(image_path: str) -> Optional[np.ndarray]:
        """
        读取图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            图像数组 (RGB格式)
        """
        try:
            # 解决中文路径问题
            img_array = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                return None
            
            # BGR -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            logger.error(f"读取图像失败: {e}")
            return None
    
    @staticmethod
    def preprocess(image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入图像 (RGB)
            target_size: 目标尺寸 (height, width)
            
        Returns:
            预处理后的图像张量 (C, H, W)
        """
        # 调整大小
        image_resized = cv2.resize(image, (target_size[1], target_size[0]))
        
        # 归一化
        image_normalized = image_resized.astype(np.float32)
        image_normalized = (image_normalized - ImagePreprocessor.MEAN) / ImagePreprocessor.STD
        
        # HWC -> CHW
        image_tensor = np.transpose(image_normalized, (2, 0, 1))
        
        return image_tensor
    
    @staticmethod
    def split_large_image(image: np.ndarray, patch_size: int = 512, overlap: int = 64) -> Tuple[list, Tuple[int, int]]:
        """
        将大图像分割成小块
        
        Args:
            image: 输入图像
            patch_size: 块大小
            overlap: 重叠区域大小
            
        Returns:
            (patches, original_size)
            - patches: 图像块列表，每个元素为 (patch, x, y)
            - original_size: 原始图像尺寸
        """
        h, w = image.shape[:2]
        patches = []
        
        # 计算步长
        step = patch_size - overlap
        
        # 生成图像块
        y = 0
        while y < h:
            x = 0
            while x < w:
                # 计算块的位置
                patch_y1 = y
                patch_y2 = min(y + patch_size, h)
                patch_x1 = x
                patch_x2 = min(x + patch_size, w)
                
                # 提取块
                patch = image[patch_y1:patch_y2, patch_x1:patch_x2]
                
                # 如果块太小，进行填充
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    padded_patch = np.zeros((patch_size, patch_size, 3), dtype=image.dtype)
                    padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                    patch = padded_patch
                
                patches.append((patch, x, y))
                x += step
            y += step
        
        return patches, (h, w)
    
    @staticmethod
    def get_padding_info(image: np.ndarray, patch_size: int = 512) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        获取图像填充信息
        
        Args:
            image: 输入图像
            patch_size: 块大小
            
        Returns:
            (padded_image, padding_info)
        """
        h, w = image.shape[:2]
        
        # 计算需要填充的大小
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        
        # 填充图像
        padded_image = np.pad(
            image,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode='constant',
            constant_values=0
        )
        
        return padded_image, (0, pad_h, 0, pad_w)
