"""
대안 XAI 기법 (SHAP, LIME) - RQ3 비교 실험용
"""
import numpy as np
import shap
import lime
from lime import lime_image
from typing import Tuple
import cv2

class AlternativeXAI:
    """대안 XAI 기법 클래스"""
    
    def __init__(self, model, preprocess_fn):
        """
        Args:
            model: Keras 모델
            preprocess_fn: 이미지 전처리 함수
        """
        self.model = model
        self.preprocess_fn = preprocess_fn
    
    def generate_shap_explanation(self, image: np.ndarray, 
                                  background_samples: np.ndarray = None) -> np.ndarray:
        """
        SHAP 설명 생성
        
        Args:
            image: 입력 이미지
            background_samples: 배경 샘플 (선택)
            
        Returns:
            SHAP 값 배열
        """
        # 이미지 전처리
        if len(image.shape) == 4:
            image = image[0]
        
        # SHAP Explainer 생성
        if background_samples is None:
            # 간단한 마스크 explainer 사용
            explainer = shap.maskers.Image("inpaint_telea", image.shape)
        else:
            explainer = shap.Explainer(self.model, background_samples)
        
        # 설명 생성
        shap_values = explainer(np.expand_dims(image, axis=0))
        
        return shap_values.values[0] if hasattr(shap_values, 'values') else shap_values
    
    def generate_lime_explanation(self, image: np.ndarray, 
                                  num_features: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        LIME 설명 생성
        
        Args:
            image: 입력 이미지
            num_features: 특징 개수
            
        Returns:
            (설명 마스크, 설명 점수) 튜플
        """
        # 이미지 전처리
        if len(image.shape) == 4:
            image = image[0]
        
        # LIME Explainer 생성
        explainer = lime_image.LimeImageExplainer()
        
        # 설명 생성
        explanation = explainer.explain_instance(
            image.astype(np.uint8),
            self.model.predict,
            top_labels=1,
            hide_color=0,
            num_samples=1000
        )
        
        # 마스크 생성
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=num_features,
            hide_rest=False
        )
        
        return mask, explanation.local_exp[explanation.top_labels[0]]
    
    def get_attention_regions_from_shap(self, shap_values: np.ndarray, 
                                       threshold: float = 0.5) -> np.ndarray:
        """SHAP 값에서 주의 영역 추출"""
        # SHAP 값을 정규화
        shap_normalized = (shap_values - shap_values.min()) / (shap_values.max() - shap_values.min() + 1e-8)
        
        # 임계값 기반 마스크
        attention_mask = (shap_normalized > threshold).astype(np.uint8)
        
        return attention_mask
    
    def get_attention_regions_from_lime(self, mask: np.ndarray) -> np.ndarray:
        """LIME 마스크에서 주의 영역 추출"""
        return mask.astype(np.uint8)
