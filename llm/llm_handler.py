"""
LLM 통합 모듈
"""
import os
import time
from typing import Dict, Optional, Tuple
from openai import OpenAI
from anthropic import Anthropic
from utils.encoding import EncodingProtocol

class LLMHandler:
    """LLM 핸들러 클래스"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", 
                 use_encoding: bool = True, encoding_map: Optional[Dict] = None):
        """
        Args:
            provider: LLM 제공자 ("openai", "anthropic", "local")
            model: 모델 이름
            use_encoding: 부호화 규약 사용 여부
            encoding_map: 부호화 규약 맵
        """
        self.provider = provider
        self.model = model
        self.use_encoding = use_encoding
        
        # 부호화 규약 초기화
        if encoding_map:
            self.encoding = EncodingProtocol(encoding_map)
        else:
            self.encoding = None
        
        # 클라이언트 초기화
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            self.client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"지원하지 않는 제공자: {provider}")
    
    def generate_explanation(self, prediction: str, probability: float, 
                            attention_regions: Optional[str] = None,
                            scenario_context: Optional[str] = None) -> Tuple[str, float]:
        """
        설명 생성
        
        Args:
            prediction: 예측 결과 ("충돌위험" 또는 "정상")
            probability: 예측 확률
            attention_regions: 주의 영역 설명 (선택)
            scenario_context: 시나리오 컨텍스트 (선택)
            
        Returns:
            (설명 텍스트, 생성 시간) 튜플
        """
        start_time = time.time()
        
        # 프롬프트 생성
        prompt = self._create_prompt(prediction, probability, attention_regions, scenario_context)
        
        # LLM 호출
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 항공기 자율비행 시스템의 결정을 설명하는 전문가입니다. 간결하고 명확하게 설명하세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            explanation = response.choices[0].message.content
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                temperature=0.7,
                system="당신은 항공기 자율비행 시스템의 결정을 설명하는 전문가입니다. 간결하고 명확하게 설명하세요.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            explanation = response.content[0].text
        
        generation_time = time.time() - start_time
        
        # 부호화 규약 적용
        if self.use_encoding and self.encoding:
            explanation = self.encoding.decode(explanation)
        
        return explanation, generation_time
    
    def _create_prompt(self, prediction: str, probability: float,
                      attention_regions: Optional[str] = None,
                      scenario_context: Optional[str] = None) -> str:
        """프롬프트 생성"""
        prompt = f"항공기 자율비행 시스템이 이미지를 분석한 결과:\n"
        prompt += f"- 예측: {prediction}\n"
        prompt += f"- 확률: {probability:.2%}\n"
        
        if attention_regions:
            prompt += f"- 주의 영역: {attention_regions}\n"
        
        if scenario_context:
            prompt += f"- 상황: {scenario_context}\n"
        
        prompt += "\n위 결과를 바탕으로 파일럿에게 간결하게 설명하세요. "
        prompt += "긴급한 상황이라면 핵심만 전달하세요."
        
        # 부호화 규약 추가
        if self.use_encoding and self.encoding:
            prompt = self.encoding.create_prompt_with_encoding(prompt)
        
        return prompt
    
    def extract_keywords(self, explanation: str) -> Dict[str, list]:
        """
        설명에서 키워드 추출
        
        Args:
            explanation: 설명 텍스트
            
        Returns:
            {"positive": [...], "negative": [...]} 딕셔너리
        """
        # 긍정 키워드 (정답 키워드)
        positive_keywords = ["충돌", "위험", "회피", "장애물", "긴급", "주의"]
        
        # 부정 키워드 (금지 키워드)
        negative_keywords = ["안전", "정상", "문제없음", "위험없음"]
        
        found_positive = [kw for kw in positive_keywords if kw in explanation]
        found_negative = [kw for kw in negative_keywords if kw in explanation]
        
        return {
            "positive": found_positive,
            "negative": found_negative
        }
