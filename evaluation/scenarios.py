"""
시나리오 정의 모듈
"""
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Scenario:
    """시나리오 데이터 클래스"""
    name: str
    image_path: str
    expected_class: str  # "충돌위험" or "정상"
    expected_keywords: List[str]
    forbidden_keywords: List[str]
    context: str

class ScenarioManager:
    """시나리오 관리자"""
    
    def __init__(self):
        self.scenarios = []
        self.load_default_scenarios()
    
    def load_default_scenarios(self):
        """기본 시나리오 로드"""
        # 충돌 위험 시나리오
        self.scenarios.append(Scenario(
            name="collision_obstacle",
            image_path="data/scenarios/collision_obstacle.jpg",
            expected_class="충돌위험",
            expected_keywords=["충돌", "위험", "장애물", "회피"],
            forbidden_keywords=["정상", "안전", "문제없음"],
            context="전방에 장애물이 감지되었습니다."
        ))
        
        self.scenarios.append(Scenario(
            name="collision_aircraft",
            image_path="data/scenarios/collision_aircraft.jpg",
            expected_class="충돌위험",
            expected_keywords=["충돌", "위험", "항공기", "회피", "긴급"],
            forbidden_keywords=["정상", "안전"],
            context="전방에 다른 항공기가 접근 중입니다."
        ))
        
        # 정상 시나리오
        self.scenarios.append(Scenario(
            name="normal_clear",
            image_path="data/scenarios/normal_clear.jpg",
            expected_class="정상",
            expected_keywords=["정상", "안전", "문제없음"],
            forbidden_keywords=["충돌", "위험", "긴급"],
            context="시야가 맑고 장애물이 없습니다."
        ))
        
        self.scenarios.append(Scenario(
            name="normal_cloud",
            image_path="data/scenarios/normal_cloud.jpg",
            expected_class="정상",
            expected_keywords=["정상", "안전"],
            forbidden_keywords=["충돌", "위험"],
            context="구름이 있지만 장애물은 없습니다."
        ))
    
    def get_scenario(self, name: str) -> Scenario:
        """시나리오 이름으로 가져오기"""
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        raise ValueError(f"시나리오를 찾을 수 없습니다: {name}")
    
    def get_all_scenarios(self) -> List[Scenario]:
        """모든 시나리오 반환"""
        return self.scenarios
    
    def get_scenarios_by_class(self, expected_class: str) -> List[Scenario]:
        """클래스별 시나리오 반환"""
        return [s for s in self.scenarios if s.expected_class == expected_class]
