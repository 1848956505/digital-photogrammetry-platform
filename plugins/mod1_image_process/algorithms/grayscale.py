"""
灰度变换与直方图均衡化
"""
import cv2
import numpy as np
from typing import Tuple, Optional


class GrayscaleProcessor:
    """灰度处理器"""
    
    @staticmethod
    def to_gray(image: np.ndarray) -> np.ndarray:
        """转换为灰度图"""
        if len(image.shape) == 2:
            return image
        if image.shape[2] == 4:  # RGBA
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def linear_transform(image: np.ndarray, alpha: float = 1.0, beta: float = 0) -> np.ndarray:
        """
        线性灰度变换
        g(x) = alpha * f(x) + beta
        
        Args:
            image: 输入图像
            alpha: 增益（对比度）
            beta: 偏移（亮度）
        """
        gray = GrayscaleProcessor.to_gray(image)
        return cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    @staticmethod
    def log_transform(image: np.ndarray, c: float = 1.0) -> np.ndarray:
        """
        对数变换
        g(x) = c * log(1 + f(x))
        
        Args:
            image: 输入图像
            c: 常数系数
        """
        gray = GrayscaleProcessor.to_gray(image)
        # 转换为float并归一化到0-1
        img_float = gray.astype(np.float32) / 255.0
        # 对数变换
        result = c * np.log1p(img_float)
        # 归一化回0-255
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result.astype(np.uint8)
    
    @staticmethod
    def exp_transform(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        指数/伽马变换
        g(x) = f(x)^gamma
        
        Args:
            image: 输入图像
            gamma: 伽马值
        """
        gray = GrayscaleProcessor.to_gray(image)
        # 归一化到0-1
        img_float = gray.astype(np.float32) / 255.0
        # 指数变换
        result = np.power(img_float, gamma)
        # 归一化回0-255
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result.astype(np.uint8)
    
    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """
        直方图均衡化
        
        Args:
            image: 输入图像
        """
        gray = GrayscaleProcessor.to_gray(image)
        return cv2.equalizeHist(gray)
    
    @staticmethod
    def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        自适应直方图均衡化 (CLAHE)
        
        Args:
            image: 输入图像
            clip_limit: 对比度限制
            tile_size: 网格大小
        """
        gray = GrayscaleProcessor.to_gray(image)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(gray)


class ResampleProcessor:
    """影像重采样处理器"""
    
    @staticmethod
    def resize(image: np.ndarray, scale: float, method: str = 'bilinear') -> np.ndarray:
        """
        图像重采样/缩放
        
        Args:
            image: 输入图像
            scale: 缩放比例
            method: 插值方法
                - 'nearest': 最近邻插值
                - 'bilinear': 双线性插值
                - 'bicubic': 双三次插值
                - 'lanczos': LANCZOS插值
        """
        methods = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        inter_method = methods.get(method, cv2.INTER_LINEAR)
        
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        
        return cv2.resize(image, (new_width, new_height), interpolation=inter_method)
    
    @staticmethod
    def resize_to_size(image: np.ndarray, width: int, height: int, method: str = 'bilinear') -> np.ndarray:
        """调整到指定尺寸"""
        methods = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        inter_method = methods.get(method, cv2.INTER_LINEAR)
        return cv2.resize(image, (width, height), interpolation=inter_method)
    
    @staticmethod
    def resize_by_factor(image: np.ndarray, fx: float, fy: float, method: str = 'bilinear') -> np.ndarray:
        """按因子缩放"""
        methods = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        inter_method = methods.get(method, cv2.INTER_LINEAR)
        return cv2.resize(image, None, fx=fx, fy=fy, interpolation=inter_method)