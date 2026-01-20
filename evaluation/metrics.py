"""
평가 지표 산출 모듈
RQ1-RQ4에 대한 정량적 평가
"""
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class EvaluationMetrics:
    """평가 지표 클래스"""
    
    def __init__(self, keywords_file: Optional[str] = None):
        """
        Args:
            keywords_file: 키워드 정의 파일 경로 (JSON)
        """
        self.keywords_file = keywords_file
        self.positive_keywords = []
        self.negative_keywords = []
        self.load_keywords()
        
        # 평가 결과 저장
        self.results = {
            "rq1": [],  # 정확성 평가
            "rq2": [],  # 실시간성 평가
            "rq3": [],  # 구성 요소 기여도
            "rq4": []   # 사용자 친화성
        }
    
    def load_keywords(self):
        """키워드 로드"""
        if self.keywords_file and Path(self.keywords_file).exists():
            with open(self.keywords_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.positive_keywords = data.get("positive", [])
                self.negative_keywords = data.get("negative", [])
        else:
            # 기본 키워드
            self.positive_keywords = ["충돌", "위험", "회피", "장애물", "긴급", "주의", "회피필요"]
            self.negative_keywords = ["안전", "정상", "문제없음", "위험없음", "정상비행"]
    
    def evaluate_rq1(self, explanation: str, expected_class: str, 
                     scenario_type: str = "collision") -> Dict:
        """
        RQ1 평가: 정확한 설명 제공 여부
        
        Args:
            explanation: 생성된 설명
            expected_class: 예상 클래스 ("충돌위험" or "정상")
            scenario_type: 시나리오 타입
            
        Returns:
            평가 결과 딕셔너리
        """
        # 키워드 매칭
        found_positive = [kw for kw in self.positive_keywords if kw in explanation]
        found_negative = [kw for kw in self.negative_keywords if kw in explanation]
        
        # 정확성 계산
        if expected_class == "충돌위험":
            # 긍정 키워드가 있어야 하고, 부정 키워드는 없어야 함
            has_positive = len(found_positive) > 0
            has_negative = len(found_negative) > 0
            accuracy = 1.0 if (has_positive and not has_negative) else 0.0
            keyword_score = len(found_positive) / max(len(self.positive_keywords), 1)
        else:
            # 정상인 경우 부정 키워드가 있어야 하고, 긍정 키워드는 없어야 함
            has_positive = len(found_positive) > 0
            has_negative = len(found_negative) > 0
            accuracy = 1.0 if (has_negative and not has_positive) else 0.0
            keyword_score = len(found_negative) / max(len(self.negative_keywords), 1)
        
        result = {
            "explanation": explanation,
            "expected_class": expected_class,
            "found_positive_keywords": found_positive,
            "found_negative_keywords": found_negative,
            "accuracy": accuracy,
            "keyword_score": keyword_score,
            "scenario_type": scenario_type
        }
        
        self.results["rq1"].append(result)
        return result
    
    def evaluate_rq2(self, event_time: float, tts_start: float, 
                     tts_end: float, threshold: float = 2.0) -> Dict:
        """
        RQ2 평가: 실시간성
        
        Args:
            event_time: 이벤트 발생 시간
            tts_start: TTS 시작 시간
            tts_end: TTS 종료 시간
            threshold: 허용 시간 임계값 (초)
            
        Returns:
            평가 결과 딕셔너리
        """
        # 응답 시간 (이벤트 발생 ~ TTS 시작)
        response_time = tts_start - event_time
        
        # 전체 처리 시간 (이벤트 발생 ~ TTS 종료)
        total_time = tts_end - event_time
        
        # TTS 재생 시간
        tts_duration = tts_end - tts_start
        
        # 실시간성 만족 여부
        is_realtime = response_time <= threshold
        
        result = {
            "response_time": response_time,
            "total_time": total_time,
            "tts_duration": tts_duration,
            "is_realtime": is_realtime,
            "threshold": threshold
        }
        
        self.results["rq2"].append(result)
        return result
    
    def evaluate_rq3(self, baseline_time: float, variant_time: float,
                     variant_name: str) -> Dict:
        """
        RQ3 평가: 구성 요소 기여도
        
        Args:
            baseline_time: 기준 시스템 처리 시간
            variant_time: 변형 시스템 처리 시간
            variant_name: 변형 이름 (예: "no_encoding", "shap", "claude")
            
        Returns:
            평가 결과 딕셔너리
        """
        speedup = baseline_time / variant_time if variant_time > 0 else 0
        improvement = ((baseline_time - variant_time) / baseline_time * 100) if baseline_time > 0 else 0
        
        result = {
            "variant_name": variant_name,
            "baseline_time": baseline_time,
            "variant_time": variant_time,
            "speedup": speedup,
            "improvement_percent": improvement
        }
        
        self.results["rq3"].append(result)
        return result
    
    def evaluate_rq4(self, user_ratings: Dict[str, float]) -> Dict:
        """
        RQ4 평가: 사용자 친화성 (설문지 기반)
        
        Args:
            user_ratings: 사용자 평가 딕셔너리
                예: {"clarity": 4.5, "helpfulness": 4.0, "non_intrusive": 3.5}
            
        Returns:
            평가 결과 딕셔너리
        """
        avg_rating = np.mean(list(user_ratings.values()))
        
        result = {
            "user_ratings": user_ratings,
            "average_rating": avg_rating,
            "total_criteria": len(user_ratings)
        }
        
        self.results["rq4"].append(result)
        return result
    
    def get_summary(self) -> Dict:
        """전체 평가 결과 요약"""
        summary = {}
        
        # RQ1 요약
        if self.results["rq1"]:
            rq1_results = self.results["rq1"]
            summary["rq1"] = {
                "total": len(rq1_results),
                "accuracy": np.mean([r["accuracy"] for r in rq1_results]),
                "avg_keyword_score": np.mean([r["keyword_score"] for r in rq1_results])
            }
        
        # RQ2 요약
        if self.results["rq2"]:
            rq2_results = self.results["rq2"]
            summary["rq2"] = {
                "total": len(rq2_results),
                "avg_response_time": np.mean([r["response_time"] for r in rq2_results]),
                "avg_total_time": np.mean([r["total_time"] for r in rq2_results]),
                "realtime_success_rate": np.mean([r["is_realtime"] for r in rq2_results])
            }
        
        # RQ3 요약
        if self.results["rq3"]:
            rq3_results = self.results["rq3"]
            summary["rq3"] = {
                "total_variants": len(rq3_results),
                "variants": {r["variant_name"]: {
                    "speedup": r["speedup"],
                    "improvement_percent": r["improvement_percent"]
                } for r in rq3_results}
            }
        
        # RQ4 요약
        if self.results["rq4"]:
            rq4_results = self.results["rq4"]
            summary["rq4"] = {
                "total": len(rq4_results),
                "avg_rating": np.mean([r["average_rating"] for r in rq4_results])
            }
        
        return summary
    
    def save_results(self, output_path: str):
        """결과 저장"""
        output = {
            "results": self.results,
            "summary": self.get_summary()
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"평가 결과 저장 완료: {output_path}")
