"""
빠른 테스트 스크립트
단일 이미지로 시스템 테스트
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import AutonomousFlightXAI

def quick_test(image_path: str):
    """
    단일 이미지로 빠른 테스트
    
    Args:
        image_path: 테스트할 이미지 경로
    """
    print("="*60)
    print("빠른 테스트 모드")
    print("="*60)
    
    # 시스템 초기화
    print("\n시스템 초기화 중...")
    system = AutonomousFlightXAI(
        use_encoding=True,
        use_gradcam=True,
        llm_provider="openai"
    )
    
    # 이미지 처리
    print(f"\n이미지 처리 중: {image_path}")
    result = system.process_image(image_path)
    
    # 결과 출력
    print("\n" + "="*60)
    print("처리 결과")
    print("="*60)
    print(f"예측: {result['prediction']}")
    print(f"확률: {result['probability']:.2%}")
    print(f"\n설명:\n{result['explanation']}")
    print(f"\n성능:")
    print(f"  - LLM 생성 시간: {result['llm_time']:.2f}초")
    print(f"  - 응답 시간: {result['response_time']:.2f}초")
    print(f"  - 전체 처리 시간: {result['total_time']:.2f}초")
    print(f"  - TTS 재생 시간: {result['tts_duration']:.2f}초")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="빠른 테스트")
    parser.add_argument("--image", type=str, required=True, help="테스트할 이미지 경로")
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"오류: 이미지 파일을 찾을 수 없습니다: {args.image}")
        sys.exit(1)
    
    quick_test(args.image)
