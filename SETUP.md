# 설정 가이드

## 1. 모델 파일 준비

`collision.h5` 파일을 프로젝트 루트 디렉토리에 배치하세요.

```bash
# 프로젝트 루트에 collision.h5 파일이 있어야 합니다
ls collision.h5
```

## 2. 환경 변수 설정

### Windows (PowerShell)
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
$env:ANTHROPIC_API_KEY="your-api-key-here"  # Claude 사용 시
```

### Windows (CMD)
```cmd
set OPENAI_API_KEY=your-api-key-here
set ANTHROPIC_API_KEY=your-api-key-here
```

### Linux/Mac
```bash
export OPENAI_API_KEY="your-api-key-here"
export ANTHROPIC_API_KEY="your-api-key-here"
```

## 3. 모델 레이어 확인

Grad-CAM을 사용하려면 모델의 레이어 이름을 확인해야 합니다.

```python
import tensorflow as tf
model = tf.keras.models.load_model("collision.h5")
for layer in model.layers:
    print(layer.name)
```

출력된 레이어 이름 중 적절한 컨볼루션 레이어를 선택하여 `config/config.py`의 `GRADCAM_LAYER_NAME`을 수정하세요.

## 4. 시나리오 이미지 준비

`data/scenarios/` 디렉토리에 시나리오 이미지를 준비하세요:

```
data/scenarios/
├── collision_obstacle.jpg
├── collision_aircraft.jpg
├── normal_clear.jpg
└── normal_cloud.jpg
```

## 5. 테스트 실행

### 빠른 테스트
```bash
python examples/quick_test.py --image path/to/test_image.jpg
```

### 전체 실험
```bash
python examples/run_experiment.py
```

### 커스텀 실행
```bash
# 시나리오 평가
python main.py --mode scenario

# 비교 실험
python main.py --mode comparison

# 단일 이미지
python main.py --mode single --image path/to/image.jpg
```

## 6. 문제 해결

### 모델 로드 오류
- 모델 파일 경로 확인
- TensorFlow 버전 확인 (2.10 이상)

### LLM API 오류
- API 키 확인
- 네트워크 연결 확인
- API 사용량 제한 확인

### TTS 오류
- pyttsx3: 시스템에 TTS 엔진이 설치되어 있는지 확인
- gTTS: 인터넷 연결 확인

### Grad-CAM 오류
- 레이어 이름 확인
- 모델 구조 확인
