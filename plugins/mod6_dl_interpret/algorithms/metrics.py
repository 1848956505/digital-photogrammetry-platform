"""
语义分割指标计算
用于评估分割结果的精度
"""
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """语义分割指标计算器"""
    
    def __init__(self, num_classes: int = 7):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    def reset(self):
        """重置混淆矩阵"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, prediction: np.ndarray, target: np.ndarray):
        """
        更新混淆矩阵
        
        Args:
            prediction: 预测结果 (H, W)
            target: 真实标签 (H, W)
        """
        # 忽略 -1 标签
        mask = target != -1
        
        # 更新混淆矩阵
        prediction_masked = prediction[mask]
        target_masked = target[mask]
        
        # 确保标签在有效范围内
        valid = (prediction_masked >= 0) & (prediction_masked < self.num_classes) & \
                (target_masked >= 0) & (target_masked < self.num_classes)
        
        pred_valid = prediction_masked[valid].astype(np.int64)
        target_valid = target_masked[valid].astype(np.int64)
        
        # 计算混淆矩阵
        hist = np.bincount(
            self.num_classes * target_valid + pred_valid,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += hist
    
    def get_stats(self) -> Dict[str, float]:
        """
        获取统计指标
        
        Returns:
            指标字典
        """
        hist = self.confusion_matrix
        
        # 计算基本指标
        ious = []
        precisions = []
        recalls = []
        f1_scores = []
        
        # 每个类别的指标
        for i in range(self.num_classes):
            # IoU
            intersection = hist[i, i]
            union = hist[i, :].sum() + hist[:, i].sum() - intersection
            
            if union > 0:
                iou = intersection / union
                ious.append(iou)
            else:
                ious.append(np.nan)
            
            # Precision
            tp = hist[i, i]
            fp = hist[:, i].sum() - tp
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
                precisions.append(precision)
            else:
                precisions.append(np.nan)
            
            # Recall
            fn = hist[i, :].sum() - tp
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
                recalls.append(recall)
            else:
                recalls.append(np.nan)
            
            # F1 Score
            if tp + fp + fn > 0:
                f1 = 2 * tp / (2 * tp + fp + fn)
                f1_scores.append(f1)
            else:
                f1_scores.append(np.nan)
        
        # 平均指标
        mean_iou = np.nanmean(ious)
        mean_precision = np.nanmean(precisions)
        mean_recall = np.nanmean(recalls)
        mean_f1 = np.nanmean(f1_scores)
        
        # 总体准确率
        overall_acc = hist.diagonal().sum() / hist.sum() if hist.sum() > 0 else 0
        
        return {
            'overall_accuracy': float(overall_acc),
            'mean_iou': float(mean_iou),
            'mean_precision': float(mean_precision),
            'mean_recall': float(mean_recall),
            'mean_f1': float(mean_f1),
            'class_iou': [float(x) if not np.isnan(x) else 0.0 for x in ious],
            'class_precision': [float(x) if not np.isnan(x) else 0.0 for x in precisions],
            'class_recall': [float(x) if not np.isnan(x) else 0.0 for x in recalls],
            'class_f1': [float(x) if not np.isnan(x) else 0.0 for x in f1_scores]
        }
    
    @staticmethod
    def calculate_pixel_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
        """
        计算像素准确率
        
        Args:
            prediction: 预测结果
            target: 真实标签
            
        Returns:
            像素准确率
        """
        mask = target != -1
        correct = (prediction[mask] == target[mask]).sum()
        total = mask.sum()
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def format_metrics_table(metrics: Dict[str, float], 
                            class_names: List[str] = None) -> str:
        """
        格式化指标表格
        
        Args:
            metrics: 指标字典
            class_names: 类别名称列表
            
        Returns:
            格式化的表格字符串
        """
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(metrics.get('class_iou', [])))]
        
        result = []
        result.append("=" * 60)
        result.append("语义分割评估指标")
        result.append("=" * 60)
        result.append(f"总体准确率: {metrics['overall_accuracy']:.4f}")
        result.append(f"平均 IoU:   {metrics['mean_iou']:.4f}")
        result.append(f"平均 Precision: {metrics['mean_precision']:.4f}")
        result.append(f"平均 Recall:  {metrics['mean_recall']:.4f}")
        result.append(f"平均 F1:      {metrics['mean_f1']:.4f}")
        result.append("")
        result.append("-" * 60)
        result.append(f"{'类别':<12} {'IoU':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        result.append("-" * 60)
        
        for i, name in enumerate(class_names):
            iou = metrics['class_iou'][i] if i < len(metrics['class_iou']) else 0
            prec = metrics['class_precision'][i] if i < len(metrics['class_precision']) else 0
            recall = metrics['class_recall'][i] if i < len(metrics['class_recall']) else 0
            f1 = metrics['class_f1'][i] if i < len(metrics['class_f1']) else 0
            
            result.append(f"{name:<12} {iou:<10.4f} {prec:<10.4f} {recall:<10.4f} {f1:<10.4f}")
        
        result.append("=" * 60)
        return "\n".join(result)
