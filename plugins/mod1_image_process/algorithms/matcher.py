"""
特征匹配算法
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import time


class TemplateMatcher:
    """基于灰度的模板匹配"""
    
    @staticmethod
    def match(image: np.ndarray, template: np.ndarray, method: str = 'tm_ccoeff') -> Tuple[np.ndarray, np.ndarray]:
        """
        模板匹配
        
        Args:
            image: 输入图像
            template: 模板图像
            method: 匹配方法
                - 'tm_sqdiff': 平方差匹配
                - 'tm_sqdiff_normed': 归一化平方差
                - 'tm_ccorr': 相关匹配
                - 'tm_ccorr_normed': 归一化相关
                - 'tm_ccoeff': 相关系数匹配
                - 'tm_ccoeff_normed': 归一化相关系数
        
        Returns:
            result: 匹配结果图
            min_val, max_val: 最佳匹配位置的值
        """
        methods = {
            'tm_sqdiff': cv2.TM_SQDIFF,
            'tm_sqdiff_normed': cv2.TM_SQDIFF_NORMED,
            'tm_ccorr': cv2.TM_CCORR,
            'tm_ccorr_normed': cv2.TM_CCORR_NORMED,
            'tm_ccoeff': cv2.TM_CCOEFF,
            'tm_ccoeff_normed': cv2.TM_CCOEFF_NORMED
        }
        
        method_enum = methods.get(method, cv2.TM_CCOEFF_NORMED)
        
        result = cv2.matchTemplate(image, template, method_enum)
        
        if method_enum in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            return result, min_val, min_loc
        else:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            return result, max_val, max_loc
    
    @staticmethod
    def match_multiple(image: np.ndarray, template: np.ndarray, 
                      threshold: float = 0.8, method: str = 'tm_ccorr_normed') -> List:
        """
        多模板匹配
        
        Args:
            image: 输入图像
            template: 模板图像
            threshold: 匹配阈值
            method: 匹配方法
            
        Returns:
            matches: 匹配位置列表 [(x, y), ...]
        """
        result, _, max_loc = TemplateMatcher.match(image, template, method)
        
        matches = []
        h, w = template.shape[:2]
        
        if method in ['tm_sqdiff_normed']:
            # 平方差越小越好
            locations = np.where(result <= threshold)
        else:
            # 相关系数越大越好
            locations = np.where(result >= threshold)
        
        for pt in zip(*locations[::-1]):
            matches.append((pt[0], pt[1], w, h))
        
        return matches


class BFMatcherWrapper:
    """Brute Force 匹配器包装"""
    
    def __init__(self, norm_type: str = 'norm_l2', cross_check: bool = False):
        """
        初始化BF匹配器
        
        Args:
            norm_type: 距离度量
                - 'norm_l1': L1距离 (ORB, BRIEF)
                - 'norm_l2': L2距离 (SIFT, SURF)
            cross_check: 是否交叉检查
        """
        self.norm_type = norm_type
        self.cross_check = cross_check
        self.matcher = None
    
    def create(self):
        """创建匹配器"""
        if self.norm_type == 'norm_l1':
            norm = cv2.NORM_L1
        elif self.norm_type == 'norm_hamming':
            norm = cv2.NORM_HAMMING
        elif self.norm_type == 'norm_hamming2':
            norm = cv2.NORM_HAMMING2
        else:
            norm = cv2.NORM_L2
        self.matcher = cv2.BFMatcher(norm, crossCheck=self.cross_check)
        return self.matcher
    
    def match(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> List:
        """单匹配"""
        if self.matcher is None:
            self.create()
        
        matches = self.matcher.match(descriptors1, descriptors2)
        # 按距离排序
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    
    def knn_match(self, descriptors1: np.ndarray, descriptors2: np.ndarray, 
                  k: int = 2) -> List:
        """KNN匹配"""
        if self.matcher is None:
            self.create()
        
        return self.matcher.knnMatch(descriptors1, descriptors2, k=k)
    
    def radius_match(self, descriptors1: np.ndarray, descriptors2: np.ndarray,
                     max_distance: float) -> List:
        """半径匹配"""
        if self.matcher is None:
            self.create()
        
        return self.matcher.radiusMatch(descriptors1, descriptors2, max_distance)


class FlannMatcherWrapper:
    """FLANN 匹配器包装（快速近似最近邻）"""
    
    def __init__(self, algorithm: str = 'kdtree', trees: int = 5, 
                 checks: int = 50, eps: float = 0):
        """
        初始化FLANN匹配器
        
        Args:
            algorithm: 算法 ('kdtree', 'kmeans', 'composite', 'autotuned')
            trees: KD树数量
            checks: 搜索检查次数
        """
        self.algorithm = algorithm
        self.trees = trees
        self.checks = checks
        self.eps = eps
        self.matcher = None
        self._index_params = None
        self._search_params = None
    
    def create(self, is_sift: bool = True):
        """创建匹配器"""
        if is_sift:
            # SIFT/SURF 使用 FLANN
            self._index_params = dict(algorithm=6,  # FLANN_INDEX_KDTREE
                                       trees=self.trees)
        else:
            # ORB 使用 LSH
            self._index_params = dict(algorithm=5,  # FLANN_INDEX_LSH
                                       table_number=6,
                                       key_size=12,
                                       multi_probe_level=1)
        
        self._search_params = dict(checks=self.checks, eps=self.eps)
        
        self.matcher = cv2.FlannBasedMatcher(self._index_params, self._search_params)
        return self.matcher
    
    def knn_match(self, descriptors1: np.ndarray, descriptors2: np.ndarray,
                  k: int = 2) -> List:
        """KNN匹配"""
        if self.matcher is None:
            self.create(is_sift=descriptors1.dtype == np.float32)
        
        # FLANN 需要 float32
        if descriptors1.dtype != np.float32:
            descriptors1 = descriptors1.astype(np.float32)
        if descriptors2.dtype != np.float32:
            descriptors2 = descriptors2.astype(np.float32)
        
        if len(descriptors1) < k or len(descriptors2) < k:
            return []
        
        return self.matcher.knnMatch(descriptors1, descriptors2, k=k)


class FeatureMatcher:
    """统一特征匹配接口"""
    
    def __init__(self, matcher_type: str = 'bf', **kwargs):
        """
        初始化特征匹配器
        
        Args:
            matcher_type: 匹配器类型 ('bf', 'flann')
            **kwargs: 匹配器参数
        """
        self.matcher_type = matcher_type
        self.matcher = None
        
        if matcher_type == 'bf':
            self.matcher = BFMatcherWrapper(**kwargs)
        elif matcher_type == 'flann':
            self.matcher = FlannMatcherWrapper(**kwargs)
        else:
            raise ValueError(f"Unknown matcher type: {matcher_type}")
    
    def match(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> List:
        """单匹配"""
        return self.matcher.match(descriptors1, descriptors2)
    
    def knn_match(self, descriptors1: np.ndarray, descriptors2: np.ndarray,
                  k: int = 2, ratio_threshold: float = 0.75) -> Tuple[List, List]:
        """
        KNN匹配并应用比率检验
        
        Args:
            descriptors1: 图像1的描述符
            descriptors2: 图像2的描述符
            k: K近邻数量
            ratio_threshold: Lowe比率检验阈值
            
        Returns:
            good_matches: 好的匹配列表
            all_matches: 所有匹配列表
        """
        knn_matches = self.matcher.knn_match(descriptors1, descriptors2, k=k)
        
        good_matches = []
        all_matches = []
        
        if k == 2 and len(knn_matches) > 0:
            # 应用比率检验
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
                    all_matches.append(m)
        else:
            # 没有足够匹配
            if len(knn_matches) > 0:
                all_matches = [m[0] for m in knn_matches if len(m) > 0]
        
        return good_matches, all_matches
    
    @staticmethod
    def draw_matches(image1: np.ndarray, keypoints1: List,
                    image2: np.ndarray, keypoints2: List,
                    matches: List, max_matches: int = 100) -> np.ndarray:
        """绘制匹配结果"""
        img1 = image1.copy()
        img2 = image2.copy()
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1:w1 + w2] = img2

        limit = min(len(matches), max_matches)
        inlier_mask = []
        if inlier_mask and len(inlier_mask) < limit:
            inlier_mask = inlier_mask + [True] * (limit - len(inlier_mask))

        for idx, match in enumerate(matches[:limit]):
            if match.queryIdx >= len(keypoints1) or match.trainIdx >= len(keypoints2):
                continue
            p1 = keypoints1[match.queryIdx].pt
            p2 = keypoints2[match.trainIdx].pt
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            pt2 = (int(round(p2[0] + w1)), int(round(p2[1])))
            inlier = True if not inlier_mask else bool(inlier_mask[idx])
            color = (46, 125, 50) if inlier else (198, 40, 40)
            cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt2, 3, color, -1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def draw_matches_colored(
        image1: np.ndarray,
        keypoints1: List,
        image2: np.ndarray,
        keypoints2: List,
        matches: List,
        max_matches: int = 100,
        inlier_mask: Optional[List[bool]] = None,
    ) -> np.ndarray:
        """绘制彩色匹配图，内点绿色，外点红色。"""
        img1 = image1.copy()
        img2 = image2.copy()
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1:w1 + w2] = img2

        limit = min(len(matches), max_matches)
        mask = list(inlier_mask or [])
        if mask and len(mask) < limit:
            mask = mask + [True] * (limit - len(mask))

        for idx, match in enumerate(matches[:limit]):
            if match.queryIdx >= len(keypoints1) or match.trainIdx >= len(keypoints2):
                continue
            p1 = keypoints1[match.queryIdx].pt
            p2 = keypoints2[match.trainIdx].pt
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            pt2 = (int(round(p2[0] + w1)), int(round(p2[1])))
            inlier = True if not mask else bool(mask[idx])
            color = (46, 125, 50) if inlier else (198, 40, 40)
            cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt2, 3, color, -1, cv2.LINE_AA)
        return canvas


class DLFeatureMatcher:
    """深度学习特征匹配（简化版，使用ORB作为替代）"""
    
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.matcher = FeatureMatcher(matcher_type='bf', norm_type='norm_hamming', cross_check=False)
    
    def match(self, image1: np.ndarray, image2: np.ndarray, 
              ratio_threshold: float = 0.75) -> Dict:
        """
        深度学习特征匹配（使用ORB模拟）
        
        Returns:
            dict: 匹配结果，包含匹配点、时间等
        """
        start_time = time.time()
        
        # 检测特征
        kp1, des1 = self.orb.detectAndCompute(image1, None)
        kp2, des2 = self.orb.detectAndCompute(image2, None)
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return {
                'algorithm': 'ORB (DL模拟)',
                'keypoints1': kp1,
                'keypoints2': kp2,
                'matches': [],
                'num_matches': 0,
                'time': time.time() - start_time,
                'note': '特征点不足'
            }
        
        # KNN匹配
        good_matches, all_matches = self.matcher.knn_match(
            des1,
            des2,
            k=2,
            ratio_threshold=ratio_threshold,
        )
        
        # 估算转换矩阵
        if len(good_matches) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # 使用 RANSAC 找到最佳变换
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # 过滤内点
            inliers = mask.ravel().sum() if mask is not None else 0
        else:
            inliers = 0
        
        elapsed_time = time.time() - start_time
        
        return {
            'algorithm': 'ORB (DL模拟)',
            'keypoints1': kp1,
            'keypoints2': kp2,
            'matches': good_matches,
            'num_matches': len(good_matches),
            'inliers': inliers,
            'time': elapsed_time,
            'note': f'检测到 {len(kp1)} 和 {len(kp2)} 个特征点'
        }


class CompareMatcher:
    """SIFT与深度学习对比匹配器"""
    
    def __init__(self):
        self.sift_processor = None
        self.dl_matcher = DLFeatureMatcher()
    
    def compare(self, image1: np.ndarray, image2: np.ndarray) -> Dict:
        """
        SIFT与DL特征匹配对比
        
        Returns:
            dict: 包含两种方法的匹配结果
        """
        # SIFT 匹配
        sift_result = self._sift_match(image1, image2)
        
        # DL (ORB模拟) 匹配
        dl_result = self.dl_matcher.match(image1, image2)
        
        return {
            'sift': sift_result,
            'dl': dl_result,
            'comparison': {
                'sift_time': sift_result['time'],
                'dl_time': dl_result['time'],
                'sift_matches': sift_result['num_matches'],
                'dl_matches': dl_result['num_matches'],
                'sift_inliers': sift_result.get('inliers', 0),
                'dl_inliers': dl_result.get('inliers', 0)
            }
        }
    
    def _sift_match(self, image1: np.ndarray, image2: np.ndarray) -> Dict:
        """SIFT特征匹配"""
        start_time = time.time()
        
        # 创建SIFT
        sift = cv2.SIFT_create()
        
        # 检测特征
        kp1, des1 = sift.detectAndCompute(image1, None)
        kp2, des2 = sift.detectAndCompute(image2, None)
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return {
                'algorithm': 'SIFT',
                'keypoints1': kp1,
                'keypoints2': kp2,
                'matches': [],
                'num_matches': 0,
                'time': time.time() - start_time
            }
        
        # 使用 BFMatcher（更稳定）
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # KNN匹配
        knn_matches = matcher.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
        
        # 比率检验
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        # 估算转换矩阵
        if len(good_matches) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = mask.ravel().sum() if mask is not None else 0
        else:
            inliers = 0
        
        elapsed_time = time.time() - start_time
        
        return {
            'algorithm': 'SIFT',
            'keypoints1': kp1,
            'keypoints2': kp2,
            'matches': good_matches,
            'num_matches': len(good_matches),
            'inliers': inliers,
            'time': elapsed_time
        }
    
    def draw_comparison(self, image1: np.ndarray, image2: np.ndarray, 
                       result: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """绘制对比结果"""
        # SIFT 结果
        sift_img = FeatureMatcher.draw_matches_colored(
            image1, result['sift']['keypoints1'],
            image2, result['sift']['keypoints2'],
            result['sift']['matches']
        )
        
        # DL 结果
        dl_img = FeatureMatcher.draw_matches_colored(
            image1, result['dl']['keypoints1'],
            image2, result['dl']['keypoints2'],
            result['dl']['matches']
        )
        
        return sift_img, dl_img
