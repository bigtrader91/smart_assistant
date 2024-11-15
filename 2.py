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
import webrtcvad
import struct

# 환경 변수 설정
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# 경고 메시지 필터링
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SmartAssistant:
    def __init__(self):
        # 오디오 장치 초기화
        self.initialize_audio_device()
        
        # Whisper 모델 로드
        self.whisper_model = whisper.load_model("turbo")
        
        # MeloTTS 초기화
        self.tts_model = TTS(language='KR', device='auto')
        self.speaker_ids = self.tts_model.hps.data.spk2id
        
        # 오디오 설정
        self.samplerate = 16000
        self.channels = 1
        self.dtype = np.int16
        self.audio_queue = queue.Queue()
        
        # VAD 설정
        self.vad = webrtcvad.Vad(3)
        self.frame_duration = 30
        self.buffer = deque(maxlen=int(self.samplerate * 0.5))
        
        # 녹음 상태 관리
        self.is_recording = False
        self.is_listening_for_wake = True
        self.recorded_frames = []
        self.silence_threshold = 0.03
        self.silence_duration = 0.7
        self.max_record_duration = 10
        
        # 웨이크워드 설정
        self.wake_word = "헤이 지피티"
        
    def initialize_audio_device(self):
        """오디오 장치 초기화 및 설정"""
        try:
            # 사용 가능한 모든 오디오 장치 출력
            print("\n사용 가능한 오디오 장치:")
            devices = sd.query_devices()
            for idx, device in enumerate(devices):
                print(f"Device {idx}: {device['name']}")
                print(f"  Input channels: {device['max_input_channels']}")
                print(f"  Output channels: {device['max_output_channels']}")
                print(f"  Sample rate: {device['default_samplerate']}")
                print()

            # pulse 장치 선택 (index 0)
            self.input_device = 0  # pulse 장치
            self.output_device = 0  # pulse 장치
            self.samplerate = int(devices[self.input_device]['default_samplerate'])
            
            # 장치 설정
            sd.default.device = (self.input_device, self.output_device)
            print(f"선택된 입력/출력 장치: {devices[self.input_device]['name']}")
            print(f"샘플레이트: {self.samplerate}Hz")
            
            # 테스트 녹음으로 장치 확인
            print("입력 장치 테스트 중...")
            with sd.InputStream(
                device=self.input_device,
                channels=self.channels,
                dtype=self.dtype,
                samplerate=self.samplerate,
                blocksize=int(self.samplerate * 0.1)  # 100ms 블록
            ) as stream:
                stream.read(frames=int(self.samplerate * 0.1))  # 0.1초 테스트
                print("입력 장치 테스트 성공")
                
        except Exception as e:
            print(f"오디오 장치 초기화 중 오류 발생: {e}")
            raise


    def is_speech(self, audio_chunk):
        """음성 감지"""
        try:
            raw_data = struct.pack("%dh" % len(audio_chunk), *audio_chunk)
            return self.vad.is_speech(raw_data, self.samplerate)
        except:
            return False

    def detect_silence(self, audio_data):
        """무음 감지"""
        return np.max(np.abs(audio_data)) < self.silence_threshold

    def audio_callback(self, indata, frames, time, status):
        """오디오 입력 콜백"""
        if status:
            print(status)
        
        # 버퍼에 데이터 추가
        self.buffer.extend(indata.flatten())
        
        if self.is_recording:
            self.recorded_frames.append(indata.copy())
        elif self.is_listening_for_wake:
            # 웨이크워드 감지 로직
            audio_data = np.array(list(self.buffer))
            if len(audio_data) >= self.samplerate:
                # 임시 파일로 저장
                temp_file = "temp_wake.wav"
                self.save_audio(audio_data, temp_file)
                
                # 음성 인식
                text = self.transcribe_audio(temp_file)
                os.remove(temp_file)
                
                if text and self.wake_word.lower() in text.lower():
                    print("\n웨이크워드가 감지되었습니다!")
                    self.is_listening_for_wake = False
                    self.start_recording()

    def start_recording(self):
        """녹음 시작"""
        self.is_recording = True
        self.recorded_frames = []
        self.recording_start_time = time.time()
        print("음성을 입력하세요...")

    def stop_recording(self):
        """녹음 중지"""
        self.is_recording = False
        return np.concatenate(self.recorded_frames, axis=0)

    def save_audio(self, audio_data, filename="temp_audio.wav"):
        """오디오 데이터를 파일로 저장"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes(audio_data.tobytes())
        return filename

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
                model="gpt-4-mini",
                messages=[
                    {"role": "system", "content": "당신은 도움이 되는 한국어 AI 어시스턴트입니다.답변을 할 때 영어 단어의 경우에는 한국어로 소리나는 대로 답변해주세요. ex) API = 에이피아이 apple = 애플"},
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
            
            data, fs = sf.read(output_path)
            sd.play(data, fs, device=self.output_device)
            sd.wait()
            
            os.remove(output_path)
            
        except Exception as e:
            print(f"음성 합성 오류: {e}")

    def audio_callback(self, indata, frames, time, status):
        """오디오 입력 콜백"""
        if status:
            print(status)
        
        if self.is_recording:
            self.recorded_frames.append(indata.copy())
        elif self.is_listening_for_wake:
            # 웨이크워드 감지 로직
            audio_data = indata.flatten()
            
            # 진폭이 특정 임계값을 넘을 때만 처리
            if np.max(np.abs(audio_data)) > self.silence_threshold:
                temp_file = "temp_wake.wav"
                self.save_audio(audio_data, temp_file)
                
                text = self.transcribe_audio(temp_file)
                os.remove(temp_file)
                
                if text and self.wake_word.lower() in text.lower():
                    print("\n웨이크워드가 감지되었습니다!")
                    self.is_listening_for_wake = False
                    self.start_recording()
    def process_voice_command(self):
        """음성 명령 처리"""
        try:
            with sd.InputStream(
                device=self.input_device,
                samplerate=self.samplerate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self.audio_callback,
                blocksize=int(self.samplerate * 0.05)  # 50ms 블록
            ) as stream:
                print("음성 입력 준비 완료...")
                
                while True:
                    if self.is_recording:
                        current_time = time.time()
                        recording_duration = current_time - self.recording_start_time
                        
                        if len(self.recorded_frames) > 0:
                            recent_audio = self.recorded_frames[-1]
                            if self.detect_silence(recent_audio):
                                silence_start = time.time()
                                
                                while time.time() - silence_start < self.silence_duration:
                                    if len(self.recorded_frames) > 0 and not self.detect_silence(self.recorded_frames[-1]):
                                        break
                                else:
                                    print("\n무음이 감지되어 녹음을 종료합니다.")
                                    break
                        
                        if recording_duration >= self.max_record_duration:
                            print("\n최대 녹음 시간에 도달했습니다.")
                            break
                            
                        time.sleep(0.1)
                    else:
                        time.sleep(0.1)
                
                if self.is_recording:
                    audio_data = self.stop_recording()
                    if len(audio_data) > 0:
                        audio_file = self.save_audio(audio_data)
                        
                        text = self.transcribe_audio(audio_file)
                        if text:
                            print(f"\n인식된 텍스트: {text}")
                            response = self.get_ai_response(text)
                            self.speak(response)
                        
                        os.remove(audio_file)
                    else:
                        print("녹음된 데이터가 없습니다.")
                    self.is_listening_for_wake = True
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            self.is_recording = False
            self.is_listening_for_wake = True
            return False
        
        return True

    def run(self):
        """메인 실행 루프"""
        print(f"AI 어시스턴트가 시작되었습니다. 웨이크워드 '{self.wake_word}'를 말씀해주세요...")
        
        while True:
            try:
                if not self.process_voice_command():
                    break
                    
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")

if __name__ == "__main__":
    assistant = SmartAssistant()
    assistant.run()