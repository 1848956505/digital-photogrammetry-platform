"""
结果后处理器
用于语义分割结果的后处理和可视化
"""
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ResultPostprocessor:
    """结果后处理器"""
    
    # LoveDA 类别定义（与原始数据集一致）
    CLASSES = {
        0: '背景',
        1: '建筑',
        2: '道路',
        3: '水体',
        4: '荒地',
        5: '林地',
        6: '农业用地'
    }
    
    # LoveDA 颜色映射 (BGR for OpenCV, 与原始数据集一致)
    COLORS = {
        0: (255, 255, 255),    # 背景 - 白色
        1: (0, 0, 255),        # 建筑 - 红色 (RGB: 255,0,0 → BGR: 0,0,255)
        2: (0, 255, 255),      # 道路 - 黄色 (RGB: 255,255,0 → BGR: 0,255,255)
        3: (255, 0, 0),        # 水体 - 蓝色 (RGB: 0,0,255 → BGR: 255,0,0)
        4: (183, 129, 159),    # 荒地 - 紫色 (RGB: 159,129,183 → BGR: 183,129,159)
        5: (0, 255, 0),        # 林地 - 绿色 (RGB: 0,255,0 → BGR: 0,255,0)
        6: (128, 195, 255)     # 农业用地 - 橙色 (RGB: 255,195,128 → BGR: 128,195,255)
    }
    
    @staticmethod
    def merge_patches(patches: list, original_size: Tuple[int, int], 
                     patch_size: int = 512, overlap: int = 64) -> np.ndarray:
        """
        合并分割后的图像块
        
        Args:
            patches: 图像块列表，每个元素为 (prediction, x, y)
            original_size: 原始图像尺寸 (h, w)
            patch_size: 块大小
            overlap: 重叠区域大小
            
        Returns:
            合并后的分割结果
        """
        h, w = original_size
        result = np.zeros((h, w), dtype=np.uint8)
        count_map = np.zeros((h, w), dtype=np.int32)
        
        step = patch_size - overlap
        
        for pred, x, y in patches:
            # 计算有效区域
            y1 = y
            y2 = min(y + patch_size, h)
            x1 = x
            x2 = min(x + patch_size, w)
            
            # 计算在pred中的区域
            pred_h = y2 - y1
            pred_w = x2 - x1
            
            # 合并结果（重叠区域取平均）
            result[y1:y2, x1:x2] += pred[:pred_h, :pred_w]
            count_map[y1:y2, x1:x2] += 1
        
        # 处理重叠区域
        count_map[count_map == 0] = 1
        result = result // count_map
        
        return result.astype(np.uint8)
    
    @staticmethod
    def remove_padding(prediction: np.ndarray, padding_info: Tuple[int, int, int, int]) -> np.ndarray:
        """
        移除填充
        
        Args:
            prediction: 预测结果
            padding_info: 填充信息 (top, bottom, left, right)
            
        Returns:
            移除填充后的结果
        """
        top, bottom, left, right = padding_info
        h, w = prediction.shape
        return prediction[top:h-bottom, left:w-right]
    
    @staticmethod
    def colorize(prediction: np.ndarray) -> np.ndarray:
        """
        将预测结果着色
        
        Args:
            prediction: 预测结果 (H, W)
            
        Returns:
            着色后的图像 (H, W, 3) BGR格式
        """
        h, w = prediction.shape
        color_result = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in ResultPostprocessor.COLORS.items():
            mask = prediction == class_id
            color_result[mask] = color
        
        return color_result
    
    @staticmethod
    def overlay(image: np.ndarray, prediction: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        将预测结果叠加在原图上
        
        Args:
            image: 原图 (RGB)
            prediction: 预测结果 (H, W)
            alpha: 透明度
            
        Returns:
            叠加后的图像
        """
        # 将RGB转为BGR
        if image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # 着色预测结果
        color_pred = ResultPostprocessor.colorize(prediction)
        
        # 叠加
        overlay = cv2.addWeighted(image_bgr, 1 - alpha, color_pred, alpha, 0)
        
        return overlay
    
    @staticmethod
    def calculate_class_distribution(prediction: np.ndarray) -> Dict[str, float]:
        """
        计算各类别分布
        
        Args:
            prediction: 预测结果
            
        Returns:
            类别分布字典 {类别名: 百分比}
        """
        total_pixels = prediction.size
        distribution = {}
        
        for class_id, class_name in ResultPostprocessor.CLASSES.items():
            count = np.sum(prediction == class_id)
            percentage = (count / total_pixels) * 100
            distribution[class_name] = percentage
        
        return distribution
    
    @staticmethod
    def resize_prediction(prediction: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        调整预测结果大小
        
        Args:
            prediction: 预测结果
            target_size: 目标尺寸 (h, w)
            
        Returns:
            调整大小后的预测结果
        """
        return cv2.resize(prediction, (target_size[1], target_size[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    @staticmethod
    def save_result(prediction: np.ndarray, output_path: str, colormap: bool = False) -> bool:
        """
        保存预测结果
        
        Args:
            prediction: 预测结果
            output_path: 输出路径
            colormap: 是否保存为彩色图像
            
        Returns:
            是否成功
        """
        try:
            if colormap:
                color_result = ResultPostprocessor.colorize(prediction)
                cv2.imwrite(output_path, color_result)
            else:
                cv2.imwrite(output_path, prediction)
            return True
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            return False
