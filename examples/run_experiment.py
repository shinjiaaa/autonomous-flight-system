"""
실험 실행 예제 스크립트
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import AutonomousFlightXAI
from evaluation.rq4_survey import UserSurvey
from evaluation.metrics import EvaluationMetrics

def run_full_experiment():
    """전체 실험 실행"""
    print("="*60)
    print("비전 기반 자율비행 시스템 XAI 평가 실험")
    print("="*60)
    
    # 시스템 초기화
    system = AutonomousFlightXAI(
        use_encoding=True,
        use_gradcam=True,
        llm_provider="openai"
    )
    
    # RQ1, RQ2: 시나리오 평가
    print("\n[1단계] RQ1, RQ2 평가: 시나리오 기반 평가")
    system.run_scenario_evaluation()
    
    # RQ3: 비교 실험
    print("\n[2단계] RQ3 평가: 구성 요소 기여도 비교 실험")
    system.run_comparison_experiment()
    
    # RQ4: 사용자 친화성 평가
    print("\n[3단계] RQ4 평가: 사용자 친화성 설문")
    survey = UserSurvey()
    
    # 시나리오 목록 가져오기
    scenarios = system.scenario_manager.get_all_scenarios()
    scenario_names = [s.name for s in scenarios]
    
    # 설문 진행
    ratings = survey.conduct_survey_batch(scenario_names)
    
    # 평가에 추가
    for rating in ratings:
        system.metrics.evaluate_rq4(rating)
    
    # 설문 결과 저장
    survey.save_ratings_to_file(ratings, "results/rq4_survey.json")
    
    # 최종 결과 요약
    print("\n" + "="*60)
    print("전체 평가 결과 요약")
    print("="*60)
    summary = system.metrics.get_summary()
    import json
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    
    # 전체 결과 저장
    system.metrics.save_results("results/full_evaluation.json")
    
    print("\n실험 완료!")

if __name__ == "__main__":
    run_full_experiment()
