import json
import queue
import sounddevice as sd
import whisper
import numpy as np
import threading
from melo.api import TTS
import time
import wave
import os
from dotenv import load_dotenv
import openai
import soundfile as sf
import warnings
from collections import deque
import pyaudio  # Windows에서 더 안정적인 오디오 처리를 위해 추가

# 환경 변수 설정
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# 경고 메시지 필터링
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SmartAssistant:
    def __init__(self):
        print("SmartAssistant 초기화 시작...")
        try:
            # PyAudio 초기화
            self.audio = pyaudio.PyAudio()
            
            # 오디오 설정
            self.samplerate = 16000
            self.channels = 1
            self.chunk_size = 1024
            self.format = pyaudio.paInt16
            
            # 오디오 장치 초기화
            self.initialize_audio_device()
            
            print("Whisper 모델 로딩 중...")
            self.whisper_model = whisper.load_model("base")
            print("Whisper 모델 로딩 완료")
            
            print("MeloTTS 초기화 중...")
            self.tts_model = TTS(language='KR', device='auto')
            self.speaker_ids = self.tts_model.hps.data.spk2id
            print("MeloTTS 초기화 완료")
            
            # 녹음 상태 관리
            self.is_recording = False
            self.is_listening_for_wake = True
            self.recorded_frames = []
            self.silence_threshold = 0.03
            self.silence_duration = 0.7
            self.max_record_duration = 10
            
            # 웨이크워드 설정
            self.wake_word = "헤이 지피티"
            
            print("SmartAssistant 초기화 완료")
            
        except Exception as e:
            print(f"초기화 중 오류 발생: {str(e)}")
            raise
            
    def initialize_audio_device(self):
        """윈도우용 오디오 장치 초기화 및 설정"""
        try:
            print("\n오디오 장치 초기화 시작...")
            
            # 사용 가능한 모든 오디오 장치 출력
            print("\n사용 가능한 입력 장치:")
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    print(f"Device {i}: {device_info['name']}")
                    print(f"  Input channels: {device_info['maxInputChannels']}")
                    print(f"  Sample rate: {int(device_info['defaultSampleRate'])}")
                    print()

            # 기본 입력 장치 찾기
            self.input_device_index = self.audio.get_default_input_device_info()['index']
            print(f"선택된 입력 장치: {self.audio.get_device_info_by_index(self.input_device_index)['name']}")
            
            # 테스트 녹음
            print("\n입력 장치 테스트 중...")
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.samplerate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size
            )
            
            data = stream.read(self.chunk_size)
            stream.stop_stream()
            stream.close()
            
            print("입력 장치 테스트 성공")
            print("오디오 장치 초기화 완료")
            
        except Exception as e:
            print(f"오디오 장치 초기화 중 오류 발생: {str(e)}")
            print("상세 오류 정보:")
            import traceback
            traceback.print_exc()
            raise

    def record_audio(self):
        """오디오 녹음"""
        print("녹음 시작...")
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.samplerate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        start_time = time.time()
        silence_start = None
        
        try:
            while True:
                data = stream.read(self.chunk_size)
                frames.append(data)
                
                # 현재 청크의 음량 레벨 체크
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume_norm = np.linalg.norm(audio_data) / len(audio_data)
                
                # 무음 감지
                if volume_norm < self.silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > self.silence_duration:
                        break
                else:
                    silence_start = None
                
                # 최대 녹음 시간 체크
                if time.time() - start_time > self.max_record_duration:
                    break
                
        finally:
            stream.stop_stream()
            stream.close()
        
        return b''.join(frames)

    def save_audio(self, audio_data, filename="temp_audio.wav"):
        """오디오 데이터를 파일로 저장"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.samplerate)
            wf.writeframes(audio_data)
        return filename

    def play_audio(self, filename):
        """오디오 파일 재생"""
        data, fs = sf.read(filename)
        stream = self.audio.open(
            format=self.audio.get_format_from_width(2),
            channels=1,
            rate=fs,
            output=True
        )
        
        # Float32 to Int16 변환
        audio_data = (data * 32767).astype(np.int16)
        
        # 청크 단위로 재생
        chunk_size = 1024
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            stream.write(chunk.tobytes())
        
        stream.stop_stream()
        stream.close()

    def transcribe_audio(self, audio_file):
        """Whisper를 사용하여 음성을 텍스트로 변환"""
        try:
            result = self.whisper_model.transcribe(
                audio_file,
                language="ko",
                task="transcribe"
            )
            return result["text"].strip()
        except Exception as e:
            print(f"음성 인식 오류: {e}")
            return None

    def get_ai_response(self, text):
        """OpenAI API를 사용하여 응답 생성"""
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 도움이 되는 한국어 AI 어시스턴트입니다. 답변을 할 때 영어 단어의 경우에는 한국어로 소리나는 대로 답변해주세요. ex) API = 에이피아이 apple = 애플"},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"AI 응답 생성 오류: {e}")
            return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."

    def speak(self, text):
        """TTS를 사용하여 음성 출력"""
        try:
            output_path = 'temp_response.wav'
            self.tts_model.tts_to_file(
                text, 
                self.speaker_ids['KR'], 
                output_path, 
                speed=1.2
            )
            print(f"AI 응답: {text}")
            
            self.play_audio(output_path)
            os.remove(output_path)
            
        except Exception as e:
            print(f"음성 합성 오류: {e}")

    def run(self):
        """메인 실행 루프"""
        print(f"\nAI 어시스턴트가 시작되었습니다.")
        print(f"웨이크워드 '{self.wake_word}'를 말씀해주세요...")
        
        while True:
            try:
                print("\n음성 입력을 기다리는 중...")
                audio_data = self.record_audio()
                audio_file = self.save_audio(audio_data)
                
                text = self.transcribe_audio(audio_file)
                if text:
                    print(f"\n인식된 텍스트: {text}")
                    
                    if self.is_listening_for_wake:
                        if self.wake_word.lower() in text.lower():
                            print("웨이크워드가 감지되었습니다!")
                            self.is_listening_for_wake = False
                            self.speak("네, 말씀하세요.")
                    else:
                        response = self.get_ai_response(text)
                        self.speak(response)
                        self.is_listening_for_wake = True
                
                os.remove(audio_file)
                    
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"실행 중 오류 발생: {str(e)}")
                print("상세 오류 정보:")
                import traceback
                traceback.print_exc()
                continue

    def __del__(self):
        """클래스 소멸자: PyAudio 정리"""
        if hasattr(self, 'audio'):
            self.audio.terminate()

if __name__ == "__main__":
    try:
        print("프로그램 시작...")
        assistant = SmartAssistant()
        assistant.run()
    except Exception as e:
        print(f"프로그램 실행 중 치명적 오류 발생: {str(e)}")
        print("상세 오류 정보:")
        import traceback
        traceback.print_exc()