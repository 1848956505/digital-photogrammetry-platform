"""
特征检测算法（SIFT、Harris）
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import time


class SIFTProcessor:
    """SIFT特征检测处理器"""
    
    def __init__(self, n_features: int = 0, n_octaves: int = 4, 
                 contrast_threshold: float = 0.04, edge_threshold: float = 10, 
                 sigma: float = 1.6):
        """
        初始化SIFT检测器
        
        Args:
            n_features: 关键点最大数量（0表示所有）
            n_octaves: 金字塔层数
            contrast_threshold: 对比度阈值
            edge_threshold: 边缘阈值
            sigma: 初始高斯sigma
        """
        self.n_features = n_features
        self.n_octaves = n_octaves
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        self.sift = None
    
    def create(self):
        """创建SIFT检测器"""
        self.sift = cv2.SIFT_create(
            nfeatures=self.n_features,
            nOctaveLayers=self.n_octaves,
            contrastThreshold=self.contrast_threshold,
            edgeThreshold=self.edge_threshold,
            sigma=self.sigma
        )
        return self.sift
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        检测并计算特征描述符
        
        Returns:
            keypoints: 关键点列表
            descriptors: 特征描述符
        """
        if self.sift is None:
            self.create()
        
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.sift.detectAndCompute(gray, None)
    
    def detect(self, image: np.ndarray) -> List:
        """仅检测关键点"""
        if self.sift is None:
            self.create()
        
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.sift.detect(gray, None)
    
    def compute(self, image: np.ndarray, keypoints: List) -> Tuple[List, np.ndarray]:
        """仅计算描述符"""
        if self.sift is None:
            self.create()
        
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.sift.compute(gray, keypoints)
    
    def draw_keypoints(self, image: np.ndarray, keypoints: List, color: Tuple = (0, 255, 0)) -> np.ndarray:
        """绘制关键点"""
        return cv2.drawKeypoints(image, keypoints, None, color)


class HarrisProcessor:
    """Harris角点检测处理器"""
    
    def __init__(self, block_size: int = 2, ksize: int = 3, k: float = 0.04):
        """
        初始化Harris检测器
        
        Args:
            block_size: 邻域大小
            ksize: Sobel核大小
            k: Harris角点常数
        """
        self.block_size = block_size
        self.ksize = ksize
        self.k = k
    
    def detect(self, image: np.ndarray, threshold: float = 0.01) -> Tuple[np.ndarray, List]:
        """
        Harris角点检测
        
        Args:
            image: 输入图像
            threshold: 阈值（0-1之间）
            
        Returns:
            corners: 角点响应图
            keypoints: 关键点列表
        """
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        
        # Harris角点检测
        corners = cv2.cornerHarris(gray, self.block_size, self.ksize, self.k)
        
        # 扩大角点以便显示
        corners = cv2.dilate(corners, None)
        
        # 提取关键点
        keypoints = []
        threshold_val = threshold * corners.max()
        
        for i in range(corners.shape[0]):
            for j in range(corners.shape[1]):
                if corners[i, j] > threshold_val:
                    keypoints.append(cv2.KeyPoint(j, i, 1))
        
        return corners, keypoints
    
    def detect_shi_tomasi(self, image: np.ndarray, max_corners: int = 100, 
                         quality_level: float = 0.01, min_distance: float = 10) -> List:
        """
        Shi-Tomasi角点检测（更好的版本）
        
        Args:
            image: 输入图像
            max_corners: 最大角点数量
            quality_level: 质量阈值
            min_distance: 最小距离
        """
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
        
        keypoints = []
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                keypoints.append(cv2.KeyPoint(x, y, 1))
        
        return keypoints
    
    def draw_corners(self, image: np.ndarray, corners: np.ndarray, 
                    keypoints: List, threshold: float = 0.01) -> np.ndarray:
        """绘制角点"""
        result = image.copy()
        
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        threshold_val = threshold * corners.max()
        
        # 绘制角点
        result[corners > threshold_val] = [0, 0, 255]
        
        # 用圆圈标记关键点
        for kp in keypoints:
            pt = (int(kp.pt[0]), int(kp.pt[1]))
            cv2.circle(result, pt, 3, (0, 255, 0), -1)
        
        return result


class ORBProcessor:
    """ORB特征检测处理器（快速替代SIFT）"""
    
    def __init__(self, n_features: int = 500, scale_factor: float = 1.2,
                 n_levels: int = 8, edge_threshold: int = 31):
        """
        初始化ORB检测器
        
        Args:
            n_features: 特征点最大数量
            scale_factor: 金字塔缩放因子
            n_levels: 金字塔层数
            edge_threshold: 边缘阈值
        """
        self.n_features = n_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        self.edge_threshold = edge_threshold
        self.orb = None
    
    def create(self):
        """创建ORB检测器"""
        self.orb = cv2.ORB_create(
            nfeatures=self.n_features,
            scaleFactor=self.scale_factor,
            nlevels=self.n_levels,
            edgeThreshold=self.edge_threshold
        )
        return self.orb
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """检测并计算特征描述符"""
        if self.orb is None:
            self.create()
        
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.orb.detectAndCompute(gray, None)
    
    def detect(self, image: np.ndarray) -> List:
        """仅检测关键点"""
        if self.orb is None:
            self.create()
        
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.orb.detect(gray, None)
    
    def compute(self, image: np.ndarray, keypoints: List) -> Tuple[List, np.ndarray]:
        """仅计算描述符"""
        if self.orb is None:
            self.create()
        
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.orb.compute(gray, keypoints)


class FeatureDetector:
    """统一特征检测接口"""
    
    @staticmethod
    def detect(image: np.ndarray, method: str = 'sift', **kwargs) -> Dict:
        """
        统一特征检测接口
        
        Args:
            image: 输入图像
            method: 检测方法 ('sift', 'harris', 'orb', 'shi_tomasi')
            **kwargs: 方法特定参数
            
        Returns:
            dict: 包含 keypoints, descriptors, time 等
        """
        start_time = time.time()
        
        if method.lower() == 'sift':
            processor = SIFTProcessor(**kwargs)
            keypoints, descriptors = processor.detect_and_compute(image)
            algorithm = 'SIFT'
            
        elif method.lower() == 'harris':
            # HarrisProcessor 只接收自身参数，UI 传入的 n_features 等需要过滤掉
            harris_kwargs = {}
            for key in ("block_size", "ksize", "k"):
                if key in kwargs:
                    harris_kwargs[key] = kwargs[key]
            processor = HarrisProcessor(**harris_kwargs)
            threshold = kwargs.get("threshold", 0.01)
            corners, keypoints = processor.detect(image, threshold=threshold)
            descriptors = None
            algorithm = 'Harris'
            
        elif method.lower() == 'shi_tomasi':
            processor = HarrisProcessor()
            # 将 n_features 映射为 Shi-Tomasi 的最大角点数，更符合 UI 的输入习惯
            max_corners = kwargs.pop("max_corners", kwargs.pop("n_features", 100))
            quality_level = kwargs.pop("quality_level", 0.01)
            min_distance = kwargs.pop("min_distance", 10)
            keypoints = processor.detect_shi_tomasi(
                image,
                max_corners=max_corners,
                quality_level=quality_level,
                min_distance=min_distance,
            )
            descriptors = None
            algorithm = 'Shi-Tomasi'
            
        elif method.lower() == 'orb':
            processor = ORBProcessor(**kwargs)
            keypoints, descriptors = processor.detect_and_compute(image)
            algorithm = 'ORB'
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        elapsed_time = time.time() - start_time
        
        return {
            'algorithm': algorithm,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'num_keypoints': len(keypoints),
            'time': elapsed_time
        }
