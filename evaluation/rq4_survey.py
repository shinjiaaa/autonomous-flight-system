"""
RQ4 사용자 친화성 평가를 위한 설문지 모듈
"""
import json
from typing import Dict, List
from pathlib import Path

class UserSurvey:
    """사용자 설문지 클래스"""
    
    def __init__(self):
        self.questions = {
            "clarity": "설명이 명확하고 이해하기 쉬웠나요? (1-5점)",
            "helpfulness": "설명이 결정을 이해하는 데 도움이 되었나요? (1-5점)",
            "non_intrusive": "설명이 비행에 방해가 되지 않았나요? (1-5점)",
            "timeliness": "설명이 적시에 제공되었나요? (1-5점)",
            "voice_quality": "음성 품질이 만족스러웠나요? (1-5점)",
            "overall_satisfaction": "전반적으로 시스템에 만족하시나요? (1-5점)"
        }
    
    def conduct_survey(self, scenario_name: str = None) -> Dict[str, float]:
        """
        설문지 진행
        
        Args:
            scenario_name: 시나리오 이름 (선택)
            
        Returns:
            평가 점수 딕셔너리
        """
        print("\n=== 사용자 친화성 평가 설문지 ===\n")
        if scenario_name:
            print(f"시나리오: {scenario_name}\n")
        
        ratings = {}
        
        for key, question in self.questions.items():
            while True:
                try:
                    rating = float(input(f"{question}\n점수 (1-5): "))
                    if 1.0 <= rating <= 5.0:
                        ratings[key] = rating
                        break
                    else:
                        print("1-5 사이의 값을 입력하세요.")
                except ValueError:
                    print("숫자를 입력하세요.")
        
        return ratings
    
    def conduct_survey_batch(self, scenario_names: List[str]) -> List[Dict[str, float]]:
        """
        여러 시나리오에 대한 일괄 설문
        
        Args:
            scenario_names: 시나리오 이름 리스트
            
        Returns:
            평가 점수 리스트
        """
        all_ratings = []
        
        for scenario_name in scenario_names:
            print(f"\n{'='*50}")
            ratings = self.conduct_survey(scenario_name)
            all_ratings.append(ratings)
        
        return all_ratings
    
    def load_ratings_from_file(self, file_path: str) -> List[Dict[str, float]]:
        """
        파일에서 평가 점수 로드
        
        Args:
            file_path: JSON 파일 경로
            
        Returns:
            평가 점수 리스트
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("ratings", [])
    
    def save_ratings_to_file(self, ratings: List[Dict[str, float]], file_path: str):
        """
        평가 점수를 파일에 저장
        
        Args:
            ratings: 평가 점수 리스트
            file_path: 저장할 파일 경로
        """
        output = {
            "ratings": ratings,
            "summary": self._calculate_summary(ratings)
        }
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n설문 결과 저장 완료: {file_path}")
    
    def _calculate_summary(self, ratings: List[Dict[str, float]]) -> Dict[str, float]:
        """평가 요약 계산"""
        if not ratings:
            return {}
        
        summary = {}
        for key in self.questions.keys():
            values = [r.get(key, 0) for r in ratings if key in r]
            if values:
                summary[key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        return summary
