"""
메인 실행 파일
비전 기반 자율비행 시스템의 실시간 결정 설명 시스템
"""
import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import *
from models.cnn_model import CNNModelLoader
from xai.gradcam import GradCAM
from xai.alternative_xai import AlternativeXAI
from llm.llm_handler import LLMHandler
from audio.tts import TextToSpeech
from evaluation.metrics import EvaluationMetrics
from evaluation.scenarios import ScenarioManager
from utils.encoding import EncodingProtocol

class AutonomousFlightXAI:
    """자율비행 XAI 시스템 메인 클래스"""
    
    def __init__(self, use_encoding: bool = True, use_gradcam: bool = True,
                 llm_provider: str = "openai", llm_model: str = None):
        """
        Args:
            use_encoding: 부호화 규약 사용 여부
            use_gradcam: Grad-CAM 사용 여부 (False면 SHAP/LIME)
            llm_provider: LLM 제공자
            llm_model: LLM 모델 이름
        """
        # 모델 로드
        print("CNN 모델 로드 중...")
        self.cnn_model = CNNModelLoader(MODEL_PATH, INPUT_SIZE)
        model = self.cnn_model.get_model()
        
        # XAI 초기화
        if use_gradcam:
            print("Grad-CAM 초기화 중...")
            self.xai = GradCAM(model, GRADCAM_LAYER_NAME, GRADCAM_ALPHA)
            self.xai_type = "gradcam"
        else:
            print("대안 XAI 초기화 중...")
            self.xai = AlternativeXAI(model, self.cnn_model.preprocess_image)
            self.xai_type = "alternative"
        
        # LLM 초기화
        if llm_model is None:
            llm_model = LLM_MODEL if llm_provider == "openai" else CLAUDE_MODEL
        
        print(f"LLM 초기화 중... ({llm_provider}, {llm_model})")
        self.llm = LLMHandler(
            provider=llm_provider,
            model=llm_model,
            use_encoding=use_encoding,
            encoding_map=ENCODING_MAP if use_encoding else None
        )
        
        # TTS 초기화
        print("TTS 초기화 중...")
        self.tts = TextToSpeech(TTS_ENGINE, TTS_LANGUAGE, TTS_RATE)
        
        # 평가 초기화
        self.metrics = EvaluationMetrics(KEYWORDS_FILE)
        
        # 시나리오 관리자
        self.scenario_manager = ScenarioManager()
        
        self.use_encoding = use_encoding
        self.use_gradcam = use_gradcam
    
    def process_image(self, image_path: str, scenario_context: str = None) -> Dict:
        """
        이미지 처리 및 설명 생성
        
        Args:
            image_path: 이미지 파일 경로
            scenario_context: 시나리오 컨텍스트
            
        Returns:
            처리 결과 딕셔너리
        """
        event_time = time.time()
        
        # 1. CNN 예측
        probability, prediction = self.cnn_model.predict(image_path)
        print(f"예측: {prediction} (확률: {probability:.2%})")
        
        # 2. XAI 적용
        preprocessed_image = self.cnn_model.preprocess_image(image_path)
        
        if self.xai_type == "gradcam":
            heatmap, overlayed = self.xai.visualize(preprocessed_image)
            attention_mask = self.xai.get_attention_regions(heatmap)
            attention_description = "Grad-CAM이 전방 장애물 영역에 주의를 기울이고 있습니다."
        else:
            # SHAP 또는 LIME 사용
            if hasattr(self.xai, 'generate_shap_explanation'):
                shap_values = self.xai.generate_shap_explanation(preprocessed_image)
                attention_mask = self.xai.get_attention_regions_from_shap(shap_values)
                attention_description = "SHAP 분석 결과 전방에 주의가 필요합니다."
            else:
                mask, _ = self.xai.generate_lime_explanation(preprocessed_image)
                attention_mask = self.xai.get_attention_regions_from_lime(mask)
                attention_description = "LIME 분석 결과 전방에 주의가 필요합니다."
        
        # 3. LLM 설명 생성
        llm_start = time.time()
        explanation, llm_time = self.llm.generate_explanation(
            prediction=prediction,
            probability=probability,
            attention_regions=attention_description,
            scenario_context=scenario_context
        )
        llm_end = time.time()
        
        print(f"생성된 설명: {explanation}")
        
        # 4. 음성 안내
        tts_start, tts_end = self.tts.speak(explanation)
        
        # 5. 평가 (RQ2: 실시간성)
        self.metrics.evaluate_rq2(event_time, tts_start, tts_end, TIMEOUT_THRESHOLD)
        
        return {
            "prediction": prediction,
            "probability": probability,
            "explanation": explanation,
            "llm_time": llm_time,
            "response_time": tts_start - event_time,
            "total_time": tts_end - event_time,
            "tts_duration": tts_end - tts_start
        }
    
    def run_scenario_evaluation(self, scenario_name: str = None):
        """
        시나리오 기반 평가 실행
        
        Args:
            scenario_name: 특정 시나리오 이름 (None이면 전체)
        """
        if scenario_name:
            scenarios = [self.scenario_manager.get_scenario(scenario_name)]
        else:
            scenarios = self.scenario_manager.get_all_scenarios()
        
        print(f"\n=== 시나리오 평가 시작 (총 {len(scenarios)}개) ===\n")
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n[{i}/{len(scenarios)}] 시나리오: {scenario.name}")
            print(f"예상 클래스: {scenario.expected_class}")
            
            if not Path(scenario.image_path).exists():
                print(f"경고: 이미지 파일을 찾을 수 없습니다: {scenario.image_path}")
                continue
            
            # 이미지 처리
            result = self.process_image(scenario.image_path, scenario.context)
            
            # RQ1 평가: 정확성
            rq1_result = self.metrics.evaluate_rq1(
                explanation=result["explanation"],
                expected_class=scenario.expected_class,
                scenario_type=scenario.name
            )
            
            print(f"RQ1 정확성: {rq1_result['accuracy']:.2f}")
            print(f"키워드 점수: {rq1_result['keyword_score']:.2f}")
            print(f"발견된 긍정 키워드: {rq1_result['found_positive_keywords']}")
            print(f"발견된 부정 키워드: {rq1_result['found_negative_keywords']}")
            print(f"응답 시간: {result['response_time']:.2f}초")
            print(f"전체 처리 시간: {result['total_time']:.2f}초")
        
        # 결과 요약
        summary = self.metrics.get_summary()
        print("\n=== 평가 결과 요약 ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        
        # 결과 저장
        results_dir = Path(RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        config_suffix = f"{self.xai_type}_{self.llm.provider}_{'encoding' if self.use_encoding else 'no_encoding'}"
        output_path = results_dir / f"evaluation_{config_suffix}.json"
        self.metrics.save_results(str(output_path))
    
    def run_comparison_experiment(self):
        """RQ3 비교 실험 실행"""
        print("\n=== RQ3 비교 실험 시작 ===\n")
        
        # 테스트 이미지 (첫 번째 시나리오 사용)
        test_scenario = self.scenario_manager.get_all_scenarios()[0]
        if not Path(test_scenario.image_path).exists():
            print(f"경고: 테스트 이미지를 찾을 수 없습니다: {test_scenario.image_path}")
            return
        
        # 기준 시스템 (Grad-CAM + Encoding + GPT)
        print("1. 기준 시스템 실행 중...")
        baseline_start = time.time()
        baseline_result = self.process_image(test_scenario.image_path, test_scenario.context)
        baseline_time = time.time() - baseline_start
        
        # 변형 1: 부호화 규약 없이
        print("\n2. 변형 1: 부호화 규약 없이 실행 중...")
        no_encoding_system = AutonomousFlightXAI(
            use_encoding=False,
            use_gradcam=True,
            llm_provider="openai",
            llm_model=LLM_MODEL
        )
        variant1_start = time.time()
        variant1_result = no_encoding_system.process_image(test_scenario.image_path, test_scenario.context)
        variant1_time = time.time() - variant1_start
        
        # 변형 2: Claude 사용
        print("\n3. 변형 2: Claude 모델 사용 중...")
        claude_system = AutonomousFlightXAI(
            use_encoding=True,
            use_gradcam=True,
            llm_provider="anthropic",
            llm_model=CLAUDE_MODEL
        )
        variant2_start = time.time()
        variant2_result = claude_system.process_image(test_scenario.image_path, test_scenario.context)
        variant2_time = time.time() - variant2_start
        
        # 변형 3: SHAP 사용
        print("\n4. 변형 3: SHAP 사용 중...")
        shap_system = AutonomousFlightXAI(
            use_encoding=True,
            use_gradcam=False,
            llm_provider="openai",
            llm_model=LLM_MODEL
        )
        variant3_start = time.time()
        variant3_result = shap_system.process_image(test_scenario.image_path, test_scenario.context)
        variant3_time = time.time() - variant3_start
        
        # 평가
        self.metrics.evaluate_rq3(baseline_time, variant1_time, "no_encoding")
        self.metrics.evaluate_rq3(baseline_time, variant2_time, "claude")
        self.metrics.evaluate_rq3(baseline_time, variant3_time, "shap")
        
        # 결과 출력
        print("\n=== RQ3 비교 실험 결과 ===")
        print(f"기준 시스템 시간: {baseline_time:.2f}초")
        print(f"부호화 규약 없이: {variant1_time:.2f}초 (개선: {((baseline_time - variant1_time) / baseline_time * 100):.1f}%)")
        print(f"Claude 사용: {variant2_time:.2f}초 (개선: {((baseline_time - variant2_time) / baseline_time * 100):.1f}%)")
        print(f"SHAP 사용: {variant3_time:.2f}초 (개선: {((baseline_time - variant3_time) / baseline_time * 100):.1f}%)")
        
        summary = self.metrics.get_summary()
        if "rq3" in summary:
            print(json.dumps(summary["rq3"], ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description="자율비행 XAI 시스템")
    parser.add_argument("--mode", type=str, default="scenario",
                       choices=["scenario", "comparison", "single"],
                       help="실행 모드")
    parser.add_argument("--image", type=str, help="단일 이미지 경로 (single 모드)")
    parser.add_argument("--scenario", type=str, help="특정 시나리오 이름")
    parser.add_argument("--no-encoding", action="store_true", help="부호화 규약 사용 안 함")
    parser.add_argument("--use-shap", action="store_true", help="SHAP 사용 (Grad-CAM 대신)")
    parser.add_argument("--llm-provider", type=str, default="openai",
                       choices=["openai", "anthropic"], help="LLM 제공자")
    parser.add_argument("--llm-model", type=str, help="LLM 모델 이름")
    
    args = parser.parse_args()
    
    # 시스템 초기화
    system = AutonomousFlightXAI(
        use_encoding=not args.no_encoding,
        use_gradcam=not args.use_shap,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model
    )
    
    # 모드별 실행
    if args.mode == "scenario":
        system.run_scenario_evaluation(args.scenario)
    elif args.mode == "comparison":
        system.run_comparison_experiment()
    elif args.mode == "single":
        if not args.image:
            print("오류: --image 옵션이 필요합니다.")
            return
        result = system.process_image(args.image)
        print("\n=== 처리 결과 ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
