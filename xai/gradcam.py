"""
Grad-CAM 구현 모듈
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from typing import Tuple, Optional
import matplotlib.pyplot as plt

class GradCAM:
    """Grad-CAM 클래스"""
    
    def __init__(self, model: keras.Model, layer_name: str, alpha: float = 0.4):
        """
        Args:
            model: Keras 모델
            layer_name: Grad-CAM을 적용할 레이어 이름
            alpha: 히트맵 투명도
        """
        self.model = model
        self.layer_name = layer_name
        self.alpha = alpha
        self.grad_model = self._build_grad_model()
    
    def _build_grad_model(self) -> keras.Model:
        """그래디언트 계산을 위한 모델 구성"""
        # 타겟 레이어 찾기
        target_layer = None
        for layer in self.model.layers:
            if layer.name == self.layer_name:
                target_layer = layer
                break
        
        if target_layer is None:
            raise ValueError(f"레이어를 찾을 수 없습니다: {self.layer_name}")
        
        # 입력과 출력 정의
        inputs = self.model.input
        outputs = [target_layer.output, self.model.output]
        
        # 그래디언트 모델 생성
        grad_model = keras.Model(inputs=inputs, outputs=outputs)
        
        return grad_model
    
    def generate_heatmap(self, image: np.ndarray, class_idx: int = 0) -> np.ndarray:
        """
        Grad-CAM 히트맵 생성
        
        Args:
            image: 전처리된 이미지 (배치 차원 포함)
            class_idx: 관심 클래스 인덱스
            
        Returns:
            히트맵 배열
        """
        with tf.GradientTape() as tape:
            # 타겟 레이어 출력과 모델 출력
            conv_outputs, predictions = self.grad_model(image)
            
            # 관심 클래스에 대한 예측값
            class_channel = predictions[:, class_idx]
        
        # 그래디언트 계산
        grads = tape.gradient(class_channel, conv_outputs)
        
        # 중요도 가중치 계산 (Global Average Pooling)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 가중치 적용
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # 정규화
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # 원본 이미지 크기로 리사이즈
        heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
        
        return heatmap
    
    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """
        히트맵을 원본 이미지에 오버레이
        
        Args:
            image: 원본 이미지 (0-1 범위, RGB)
            heatmap: 히트맵 배열
            
        Returns:
            오버레이된 이미지
        """
        # 이미지가 배치 차원을 가지고 있다면 제거
        if len(image.shape) == 4:
            image = image[0]
        
        # 이미지를 0-255 범위로 변환
        image = (image * 255).astype(np.uint8)
        
        # 히트맵을 컬러맵으로 변환
        heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # 오버레이
        overlayed = cv2.addWeighted(image, 1 - self.alpha, heatmap_colored, self.alpha, 0)
        
        return overlayed
    
    def get_attention_regions(self, heatmap: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        주의 영역 추출
        
        Args:
            heatmap: 히트맵 배열
            threshold: 임계값
            
        Returns:
            주의 영역 마스크
        """
        attention_mask = (heatmap > threshold).astype(np.uint8)
        return attention_mask
    
    def visualize(self, image: np.ndarray, save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Grad-CAM 시각화
        
        Args:
            image: 전처리된 이미지
            save_path: 저장 경로 (선택)
            
        Returns:
            (히트맵, 오버레이 이미지) 튜플
        """
        # 히트맵 생성
        heatmap = self.generate_heatmap(image, class_idx=0)
        
        # 오버레이
        overlayed = self.overlay_heatmap(image, heatmap)
        
        # 저장
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))
        
        return heatmap, overlayed
