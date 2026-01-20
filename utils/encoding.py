"""
부호화 규약 모듈
LLM 출력을 최소화하여 속도 향상
"""
from typing import Dict, List, Tuple
import re

class EncodingProtocol:
    """부호화 규약 클래스"""
    
    def __init__(self, encoding_map: Dict[str, str]):
        """
        Args:
            encoding_map: 키워드-부호 매핑 딕셔너리
        """
        self.encoding_map = encoding_map
        self.reverse_map = {v: k for k, v in encoding_map.items()}
    
    def encode(self, text: str) -> str:
        """
        텍스트를 부호화 규약에 따라 인코딩
        
        Args:
            text: 원본 텍스트
            
        Returns:
            인코딩된 텍스트
        """
        encoded = text
        for keyword, code in self.encoding_map.items():
            encoded = encoded.replace(keyword, code)
        return encoded
    
    def decode(self, encoded_text: str) -> str:
        """
        부호화된 텍스트를 디코딩
        
        Args:
            encoded_text: 인코딩된 텍스트
            
        Returns:
            디코딩된 텍스트
        """
        decoded = encoded_text
        for code, keyword in self.reverse_map.items():
            decoded = decoded.replace(code, keyword)
        return decoded
    
    def extract_codes(self, text: str) -> List[str]:
        """
        텍스트에서 부호 추출
        
        Args:
            text: 부호가 포함된 텍스트
            
        Returns:
            추출된 부호 리스트
        """
        codes = []
        for code in self.reverse_map.keys():
            if code in text:
                codes.append(code)
        return codes
    
    def create_prompt_with_encoding(self, base_prompt: str) -> str:
        """
        부호화 규약을 포함한 프롬프트 생성
        
        Args:
            base_prompt: 기본 프롬프트
            
        Returns:
            부호화 규약이 포함된 프롬프트
        """
        encoding_instruction = "\n\n부호화 규약:\n"
        for keyword, code in self.encoding_map.items():
            encoding_instruction += f"- {keyword} = {code}\n"
        encoding_instruction += "\n위 규약을 사용하여 간결하게 응답하세요."
        
        return base_prompt + encoding_instruction
