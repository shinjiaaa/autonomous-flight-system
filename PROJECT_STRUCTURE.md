# 프로젝트 구조

## 디렉토리 구조

```
autonomous-flight-system/
├── config/                 # 설정 파일
│   ├── __init__.py
│   └── config.py          # 전역 설정
│
├── models/                 # CNN 모델
│   ├── __init__.py
│   └── cnn_model.py        # 모델 로더 및 추론
│
├── xai/                    # 설명 가능 AI 모듈
│   ├── __init__.py
│   ├── gradcam.py          # Grad-CAM 구현
│   └── alternative_xai.py  # SHAP, LIME (비교 실험용)
│
├── llm/                    # LLM 통합
│   ├── __init__.py
│   └── llm_handler.py      # LLM 핸들러 (OpenAI, Anthropic)
│
├── audio/                  # 음성 안내
│   ├── __init__.py
│   └── tts.py              # TTS (Text-to-Speech)
│
├── evaluation/             # 평가 모듈
│   ├── __init__.py
│   ├── metrics.py          # 평가 지표 (RQ1-RQ4)
│   ├── scenarios.py        # 시나리오 정의
│   └── rq4_survey.py       # 사용자 설문지
│
├── utils/                  # 유틸리티
│   ├── __init__.py
│   └── encoding.py         # 부호화 규약
│
├── data/                   # 데이터
│   ├── keywords.json       # 키워드 정의
│   └── scenarios/          # 시나리오 이미지 (사용자 준비)
│
├── examples/               # 예제 스크립트
│   ├── __init__.py
│   ├── quick_test.py       # 빠른 테스트
│   └── run_experiment.py   # 전체 실험 실행
│
├── results/                # 결과 저장 (자동 생성)
│
├── main.py                 # 메인 실행 파일
├── requirements.txt        # 패키지 의존성
├── README.md               # 프로젝트 설명
├── SETUP.md                # 설정 가이드
└── .gitignore              # Git 무시 파일
```

## 주요 모듈 설명

### 1. CNN 모델 (`models/cnn_model.py`)
- `collision.h5` 모델 로드
- 이미지 전처리
- 충돌 위험 분류 (이진 분류)

### 2. Grad-CAM (`xai/gradcam.py`)
- 모델의 주의 영역 시각화
- 히트맵 생성 및 오버레이

### 3. LLM 핸들러 (`llm/llm_handler.py`)
- OpenAI GPT 또는 Anthropic Claude 통합
- 부호화 규약 적용
- 설명 생성

### 4. TTS (`audio/tts.py`)
- 텍스트를 음성으로 변환
- pyttsx3 또는 gTTS 지원

### 5. 평가 지표 (`evaluation/metrics.py`)
- **RQ1**: 정확성 (키워드 매칭)
- **RQ2**: 실시간성 (응답 시간)
- **RQ3**: 구성 요소 기여도 (비교 실험)
- **RQ4**: 사용자 친화성 (설문)

### 6. 부호화 규약 (`utils/encoding.py`)
- LLM 출력 최소화
- 키워드-부호 매핑
- 속도 향상

## 데이터 흐름

```
이미지 입력
    ↓
CNN 모델 예측
    ↓
Grad-CAM 시각화
    ↓
LLM 설명 생성 (부호화 규약 적용)
    ↓
음성 안내 (TTS)
    ↓
평가 지표 산출
```

## 실행 흐름

1. **시스템 초기화**: 모델, XAI, LLM, TTS 로드
2. **이미지 처리**: CNN 예측 + Grad-CAM
3. **설명 생성**: LLM이 결정 설명 생성
4. **음성 출력**: TTS로 음성 안내
5. **평가**: RQ1-RQ4 지표 계산
6. **결과 저장**: JSON 형식으로 저장

## 확장 가능성

- 새로운 XAI 기법 추가: `xai/` 디렉토리에 추가
- 새로운 LLM 제공자: `llm/llm_handler.py` 확장
- 새로운 평가 지표: `evaluation/metrics.py` 확장
- 새로운 시나리오: `evaluation/scenarios.py` 수정
