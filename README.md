# 비전 기반 자율비행 시스템의 실시간 결정 설명을 위한 Grad-CAM 및 LLM 기반 음성 안내 시스템

## 개요

인간과의 협업 하의 자율비행 시스템에서 파일럿의 신뢰를 확보하기 위한 설명 가능한 AI(XAI) 시스템입니다. Grad-CAM을 통한 시각적 설명과 LLM 기반 음성 안내를 결합하여 실시간으로 결정을 설명합니다.

## 주요 기능

- **CNN 기반 충돌 위험 분류**: 비전 센서 입력을 분석하여 충돌 위험 여부를 판단
- **Grad-CAM 시각화**: 모델이 주의를 기울이는 영역을 시각화
- **LLM 기반 설명 생성**: 자연어로 결정 이유를 설명
- **음성 안내**: 파일럿의 시각 부하를 줄이기 위한 음성 출력
- **부호화 규약**: LLM 출력을 최소화하여 실시간성 향상
- **정량적 평가**: RQ1-RQ4에 대한 평가 지표 산출

## 시스템 구조

```
autonomous-flight-system/
├── config/           # 설정 파일
├── models/          # CNN 모델 로더
├── xai/             # XAI 모듈 (Grad-CAM, SHAP, LIME)
├── llm/             # LLM 통합 모듈
├── audio/           # 음성 안내 시스템
├── evaluation/      # 평가 지표 및 시나리오
├── utils/           # 유틸리티 (부호화 규약 등)
├── data/            # 데이터 및 시나리오
└── main.py          # 메인 실행 파일
```

## 설치

### 요구사항

- Python 3.8 이상
- TensorFlow 2.10 이상
- CUDA (GPU 사용 시)

### 설치 방법

```bash
# 저장소 클론
git clone <repository-url>
cd autonomous-flight-system

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 환경 변수 설정

```bash
# OpenAI API 키 (필수)
export OPENAI_API_KEY="your-api-key"

# Anthropic API 키 (Claude 사용 시)
export ANTHROPIC_API_KEY="your-api-key"
```

## 사용 방법

### 1. 모델 준비

`collision.h5` 파일을 프로젝트 루트 디렉토리에 배치하세요.

### 2. 기본 실행

#### 시나리오 기반 평가
```bash
python main.py --mode scenario
```

#### 특정 시나리오 실행
```bash
python main.py --mode scenario --scenario collision_obstacle
```

#### 단일 이미지 처리
```bash
python main.py --mode single --image path/to/image.jpg
```

#### 비교 실험 (RQ3)
```bash
python main.py --mode comparison
```

### 3. 옵션

- `--no-encoding`: 부호화 규약 사용 안 함
- `--use-shap`: Grad-CAM 대신 SHAP 사용
- `--llm-provider`: LLM 제공자 선택 (openai, anthropic)
- `--llm-model`: 특정 LLM 모델 지정

## 연구 질문 (RQ)

### RQ1: 정확한 설명 제공 여부
- 정답 키워드/금지 키워드 식별
- 키워드 매칭 점수 계산

### RQ2: 실시간성
- 이벤트 발생부터 음성 송출 시작까지의 시간 측정
- 전체 처리 시간 측정

### RQ3: 구성 요소 기여도
- LLM 모델 변경 실험 (GPT vs Claude)
- XAI 기법 변경 실험 (Grad-CAM vs SHAP/LIME)
- 부호화 규약 유무 비교

### RQ4: 사용자 친화성
- 설문지 기반 평가 (수동 입력)

## 평가 결과

평가 결과는 `results/` 디렉토리에 JSON 형식으로 저장됩니다.

```json
{
  "results": {
    "rq1": [...],
    "rq2": [...],
    "rq3": [...],
    "rq4": [...]
  },
  "summary": {
    "rq1": {
      "accuracy": 0.95,
      "avg_keyword_score": 0.87
    },
    "rq2": {
      "avg_response_time": 1.2,
      "realtime_success_rate": 0.98
    }
  }
}
```

## 설정

`config/config.py`에서 다음 설정을 변경할 수 있습니다:

- 모델 경로 및 입력 크기
- Grad-CAM 레이어 이름
- LLM 모델 및 제공자
- 부호화 규약 매핑
- TTS 엔진 및 속도
- 평가 임계값

## 시나리오

시나리오는 `evaluation/scenarios.py`에서 정의되며, 다음을 포함합니다:

- 충돌 위험 시나리오 (장애물, 다른 항공기)
- 정상 비행 시나리오 (맑은 하늘, 구름)

새로운 시나리오를 추가하려면 `ScenarioManager` 클래스를 수정하세요.

## 주요 기여

1. **새로운 설명 가능 시스템 제안**: Grad-CAM과 LLM을 결합한 실시간 설명 시스템
2. **항공 분야 설명성을 위한 LLM 부호화 전략**: 속도 향상을 위한 부호화 규약
3. **정성적, 정량적 평가 수행**: RQ1-RQ4에 대한 종합 평가

## 제약사항

- CNN 모델의 레이어 구조에 따라 Grad-CAM 레이어 이름을 수정해야 할 수 있습니다
- LLM API 키가 필요합니다
- 실시간성은 네트워크 상태와 LLM 응답 시간에 의존합니다

## 라이선스

[라이선스 정보]

## 참고문헌

- EASA XAI 가이드라인
- Grad-CAM 논문
- 관련 연구 논문들
