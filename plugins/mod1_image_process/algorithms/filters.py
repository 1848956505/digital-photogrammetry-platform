"""
空间滤波算法
"""
import cv2
import numpy as np
from typing import Tuple, Optional


class FilterProcessor:
    """空间滤波处理器"""
    
    # ==================== 平滑滤波 ====================
    
    @staticmethod
    def mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        均值滤波
        
        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数）
        """
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.blur(gray, (kernel_size, kernel_size))
    
    @staticmethod
    def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        中值滤波
        
        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数）
        """
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.medianBlur(gray, kernel_size)
    
    @staticmethod
    def gaussian_filter(image: np.ndarray, kernel_size: int = 3, sigma: float = 0) -> np.ndarray:
        """
        高斯滤波
        
        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数）
            sigma: 高斯核标准差
        """
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    
    @staticmethod
    def bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        双边滤波（保边平滑）
        
        Args:
            image: 输入图像
            d: 邻域直径
            sigma_color: 颜色空间sigma
            sigma_space: 坐标空间sigma
        """
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    
    # ==================== 锐化滤波 ====================
    
    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """
        锐化滤波（拉普拉斯算子）
        """
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 拉普拉斯算子
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])
        
        laplacian = cv2.filter2D(gray, -1, kernel)
        result = cv2.add(gray, laplacian)
        return cv2.convertScaleAbs(result)
    
    @staticmethod
    def sharpen_enhanced(image: np.ndarray) -> np.ndarray:
        """
        增强锐化（使用更大的核）
        """
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 增强拉普拉斯算子
        kernel = np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]])
        
        laplacian = cv2.filter2D(gray, -1, kernel)
        result = cv2.add(gray, laplacian)
        return cv2.convertScaleAbs(result)
    
    @staticmethod
    def unsharp_mask(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0, threshold: int = 0) -> np.ndarray:
        """
        Unsharp Mask 锐化
        
        Args:
            image: 输入图像
            kernel_size: 高斯核大小
            sigma: 高斯核标准差
            amount: 锐化强度
            threshold: 阈值
        """
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 获取模糊版本
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
        
        # 计算锐化图像
        sharpened = float(amount + 1) * gray - float(amount) * blurred
        
        # 应用阈值
        if threshold > 0:
            low = np.where(sharpened < gray - threshold)
            high = np.where(sharpened > gray + threshold)
            sharpened[low] = gray[low]
            sharpened[high] = gray[high]
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # ==================== 边缘检测 ====================
    
    @staticmethod
    def sobel_edge(image: np.ndarray) -> np.ndarray:
        """Sobel边缘检测"""
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = np.clip(sobel, 0, 255).astype(np.uint8)
        
        return sobel
    
    @staticmethod
    def canny_edge(image: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> np.ndarray:
        """Canny边缘检测"""
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, threshold1, threshold2)
    
    @staticmethod
    def laplacian_edge(image: np.ndarray) -> np.ndarray:
        """拉普拉斯边缘检测"""
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.absolute(laplacian)
        laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)
        
        return laplacian
    
    # ==================== 自定义卷积 ====================
    
    @staticmethod
    def custom_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        自定义卷积滤波
        
        Args:
            image: 输入图像
            kernel: 自定义卷积核
        """
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.filter2D(gray, -1, kernel)
    
    @staticmethod
    def create_box_kernel(size: int) -> np.ndarray:
        """创建盒式卷积核"""
        return np.ones((size, size), np.float32) / (size * size)
    
    @staticmethod
    def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        """创建高斯卷积核"""
        ax = cv2.getGaussianKernel(size, sigma)
        kernel = ax * ax.T
        return kernel / kernel.sum()


class MorphologicalProcessor:
    """形态学处理"""
    
    @staticmethod
    def erode(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
        """腐蚀"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.erode(image, kernel, iterations=iterations)
    
    @staticmethod
    def dilate(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
        """膨胀"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.dilate(image, kernel, iterations=iterations)
    
    @staticmethod
    def open(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """开运算（先腐蚀后膨胀）"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    @staticmethod
    def close(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """闭运算（先膨胀后腐蚀）"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)