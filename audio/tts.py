"""
음성 안내 시스템 (TTS)
"""
import time
import pyttsx3
from gtts import gTTS
import io
from typing import Optional, Tuple
import os

class TextToSpeech:
    """TTS 클래스"""
    
    def __init__(self, engine: str = "pyttsx3", language: str = "ko", rate: int = 150):
        """
        Args:
            engine: TTS 엔진 ("pyttsx3" or "gtts")
            language: 언어 코드
            rate: 말하기 속도
        """
        self.engine_type = engine
        self.language = language
        self.rate = rate
        
        if engine == "pyttsx3":
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            # 한국어 음성 설정 시도
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'korean' in voice.name.lower() or 'ko' in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
        else:
            self.engine = None
    
    def speak(self, text: str) -> Tuple[float, float]:
        """
        텍스트를 음성으로 변환하여 재생
        
        Args:
            text: 음성으로 변환할 텍스트
            
        Returns:
            (시작 시간, 종료 시간) 튜플 (초 단위)
        """
        start_time = time.time()
        
        if self.engine_type == "pyttsx3":
            self.engine.say(text)
            self.engine.runAndWait()
        elif self.engine_type == "gtts":
            # gTTS는 파일로 저장 후 재생
            tts = gTTS(text=text, lang=self.language, slow=False)
            temp_file = "temp_audio.mp3"
            tts.save(temp_file)
            
            # 재생 (시스템에 따라 다름)
            # Windows의 경우
            if os.name == 'nt':
                os.system(f"start {temp_file}")
            else:
                os.system(f"mpg123 {temp_file}")
            
            # 파일 삭제는 백그라운드에서
            time.sleep(0.5)  # 재생 시작 대기
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        end_time = time.time()
        
        return start_time, end_time
    
    def get_duration_estimate(self, text: str) -> float:
        """
        텍스트의 예상 재생 시간 추정
        
        Args:
            text: 텍스트
            
        Returns:
            예상 재생 시간 (초)
        """
        # 한국어 기준: 평균 3-4자/초
        char_count = len(text)
        estimated_duration = char_count / 3.5  # 초
        
        return estimated_duration
