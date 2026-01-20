"""
CNN 모델 로더 및 추론 모듈
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
import cv2
from typing import Tuple, Optional
import os

class CNNModelLoader:
    """CNN 모델 로더 클래스"""
    
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            model_path: 모델 파일 경로 (.h5)
            input_size: 입력 이미지 크기 (width, height)
        """
        self.model_path = model_path
        self.input_size = input_size
        self.model = None
        self.load_model()
    
    def load_model(self):
        """모델 로드"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"모델 로드 완료: {self.model_path}")
        except Exception as e:
            raise Exception(f"모델 로드 실패: {str(e)}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        이미지 전처리
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            전처리된 이미지 배열
        """
        # 이미지 읽기
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 리사이즈
        img = cv2.resize(img, self.input_size)
        
        # 정규화 (0-1 범위)
        img = img.astype(np.float32) / 255.0
        
        # 배치 차원 추가
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image_path: str) -> Tuple[float, str]:
        """
        이미지에 대한 예측 수행
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            (확률, 클래스) 튜플
        """
        # 이미지 전처리
        preprocessed = self.preprocess_image(image_path)
        
        # 예측
        prediction = self.model.predict(preprocessed, verbose=0)
        probability = float(prediction[0][0])
        
        # 이진 분류: 0 = 정상, 1 = 충돌 위험
        class_name = "충돌위험" if probability > 0.5 else "정상"
        
        return probability, class_name
    
    def get_model(self) -> keras.Model:
        """모델 객체 반환"""
        return self.model
    
    def get_layer_by_name(self, layer_name: str) -> Optional[keras.layers.Layer]:
        """
        레이어 이름으로 레이어 가져오기
        
        Args:
            layer_name: 레이어 이름
            
        Returns:
            레이어 객체 또는 None
        """
        try:
            return self.model.get_layer(layer_name)
        except:
            # 레이어 이름으로 찾기 시도
            for layer in self.model.layers:
                if layer.name == layer_name:
                    return layer
            return None
