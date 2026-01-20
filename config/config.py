"""
설정 파일
"""
import os

# 모델 경로
MODEL_PATH = "collision.h5"
INPUT_SIZE = (224, 224)  # CNN 모델 입력 크기

# Grad-CAM 설정
GRADCAM_LAYER_NAME = "conv2d_2"  # 모델에 따라 변경 필요
GRADCAM_ALPHA = 0.4

# LLM 설정
LLM_PROVIDER = "openai"  # "openai" or "anthropic" or "local"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = "gpt-4o-mini"  # 기본 모델
CLAUDE_MODEL = "claude-3-haiku-20240307"  # 비교 실험용

# 부호화 규약 설정
USE_ENCODING = True  # 부호화 규약 사용 여부
ENCODING_MAP = {
    "중요": "1",
    "보통": "2",
    "낮음": "3",
    "충돌위험": "C",
    "회피": "A",
    "정상": "N",
    "긴급": "E"
}

# 음성 설정
TTS_ENGINE = "pyttsx3"  # "pyttsx3" or "gtts"
TTS_LANGUAGE = "ko"
TTS_RATE = 150  # 말하기 속도

# 평가 설정
SCENARIO_DIR = "data/scenarios"
RESULTS_DIR = "results"
KEYWORDS_FILE = "data/keywords.json"

# 실시간성 측정
TIMEOUT_THRESHOLD = 2.0  # 초 (음성 송출 시작까지 허용 시간)
