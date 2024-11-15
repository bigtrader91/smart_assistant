import json
import queue
import sounddevice as sd
import whisper
import numpy as np
import threading
import requests
import time
import wave
import os
from dotenv import load_dotenv
import openai
import soundfile as sf
import warnings
from collections import deque
import pyaudio
import keyboard
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

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

            # TTS API 설정
            self.tts_url = "http://127.0.0.1:1234/synthesize"
            self.tts_headers = {"Content-Type": "application/json"}

            # 오디오 장치 초기화
            self.initialize_audio_device()

            print("Whisper 모델 로딩 중...")
            self.whisper_model = whisper.load_model("turbo") # turbo라는 이름도 존재해. 변경하지마
            print("Whisper 모델 로딩 완료")

            # 녹음 상태 관리
            self.is_recording = False
            self.is_speaking = False
            self.recorded_frames = []
            self.first_recording = True
            
            # Thread Pool 초기화
            self.executor = ThreadPoolExecutor(max_workers=3)

            # 대화 히스토리 초기화
            self.message_history = [
                {
                    "role": "system",
                    "content": (
                        "당신은 도움이 되는 한국의 김비서입니다. 아래 사항을 반드시 준수하세요."
                        "1. 답변을 할 때 영어 단어의 경우에는 한국어로 소리나는 대로 답변한다."
                        "2. 항상 존댓말을 사용한다."
                        "3. 짧은 문장으로 응답해서 줄바꿈이 이루어지도록 작성한다."
                    )
                }
            ]
            # 시작 멘트 설정
            self.greeting_message = "안녕하세요. 저는 김비서입니다. 무엇을 도와드릴까요?"
            self.greeting_audio_path = "greeting.wav"
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
        """엔터 키로 제어되는 오디오 녹음"""
        if self.first_recording:
            print("\n엔터 키를 눌러 녹음을 시작하세요...")
            self.first_recording = False

        keyboard.wait('enter')
        print("녹음 시작... (엔터 키를 다시 누르면 녹음이 종료됩니다)")

        # 엔터 키가 RELEASE될 때까지 기다립니다.
        while keyboard.is_pressed('enter'):
            pass

        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.samplerate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk_size
        )

        frames = []
        self.is_recording = True

        while self.is_recording:
            # 녹음 중에 엔터 키가 눌렸는지 체크합니다.
            if keyboard.is_pressed('enter'):
                self.is_recording = False
                # 엔터 키 RELEASE를 기다려 다음 녹음에 영향을 주지 않도록 합니다.
                while keyboard.is_pressed('enter'):
                    pass
                break
            data = stream.read(self.chunk_size)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        print("녹음 완료\n")

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
        try:
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

        except Exception as e:
            print(f"오디오 재생 중 오류 발생: {e}")
            print("상세 오류 정보:")
            import traceback
            traceback.print_exc()

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

            
    def maintain_message_history(self, max_length=20):
        """대화 히스토리의 길이를 유지"""
        while len(self.message_history) > max_length:
            self.message_history.pop(1)  # 시스템 메시지는 유지하고 사용자/어시스턴트 메시지 제거
            
    def get_ai_response(self, text):
        """OpenAI API를 사용하여 응답 생성 (대화 히스토리 포함) openai 1.0 버전이므로 openai.chat.completions.create 이런식으로 사용함"""
        try:
            # 사용자 메시지를 히스토리에 추가
            self.message_history.append({"role": "user", "content": text})
            
            # 히스토리 길이 유지
            self.maintain_message_history()
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",  # 이 모델만 사용. 변경하지마
                messages=self.message_history
            )
            response_text = response.choices[0].message.content

            # 어시스턴트 응답을 히스토리에 추가
            self.message_history.append({"role": "assistant", "content": response_text})

            return response_text

        except Exception as e:
            print(f"AI 응답 생성 오류: {e}")
            return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."
   
    async def process_tts(self, text, index, total):
        """TTS 처리 및 재생을 위한 비동기 함수"""
        try:
            self.is_speaking = True
            output_path = f'temp_response_{index}.wav'
            
            # TTS API 호출
            tts_data = {
                "language": "KR",
                "text": text,
                "speed": 1.2
            }
            
            # TTS API 호출을 ThreadPool에서 실행
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: requests.post(self.tts_url, headers=self.tts_headers, json=tts_data)
            )
            
            if response.status_code == 200:
                # 응답 데이터를 파일로 저장
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                print(f"AI 응답 ({index+1}/{total}): {text}")
                
                # 오디오 재생을 ThreadPool에서 실행
                await loop.run_in_executor(self.executor, self.play_audio, output_path)
                os.remove(output_path)
            else:
                print(f"TTS API 오류: {response.status_code}")
            
        except Exception as e:
            print(f"음성 합성 오류: {e}")
        finally:
            self.is_speaking = False

    def split_into_sentences(self, text):
        """텍스트를 줄바꿈 단위로 분할"""
        # 줄바꿈으로 문장 분리 및 빈 줄 제거
        return [sent.strip() for sent in text.split('\n') if sent.strip()]

    async def speak_async(self, text):
        """문장 단위로 분할하여 비동기적으로 TTS 처리"""
        # 인사 메시지인 경우 미리 생성된 오디오 파일 재생
        if text == self.greeting_message and os.path.exists(self.greeting_audio_path):
            print(f"AI 응답 (1/1): {text}")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self.play_audio, self.greeting_audio_path)
            return

        sentences = self.split_into_sentences(text)
        total_sentences = len(sentences)

        if total_sentences > 3:
            print(f"\n긴 응답을 처리합니다. 총 {total_sentences}개의 문장이 순차적으로 재생됩니다.")

        for i, sentence in enumerate(sentences):
            await self.process_tts(sentence, i, total_sentences)
            await asyncio.sleep(0.2)  # 문장 사이 약간의 간격


    def speak(self, text):
        """동기 방식의 speak 메소드"""
        asyncio.run(self.speak_async(text))
        
                
    async def run_async(self):
        """비동기 실행 루프"""
        print("\nAI 어시스턴트가 시작되었습니다.")
        print("대화를 시작하려면 엔터 키를 눌러주세요.")

        # 시작 멘트 재생 (TTS API 대신 오디오 파일 사용)
        if os.path.exists(self.greeting_audio_path):
            print(f"AI 응답 (1/1): {self.greeting_message}")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self.play_audio, self.greeting_audio_path)
        else:
            print("greeting.wav 파일을 찾을 수 없습니다. TTS API를 사용하여 인사 메시지를 생성합니다.")
            await self.speak_async(self.greeting_message)

        while True:
            try:
                if self.is_speaking:
                    await asyncio.sleep(0.1)
                    continue

                # 녹음은 동기 방식으로 유지
                audio_data = self.record_audio()
                audio_file = self.save_audio(audio_data)

                text = self.transcribe_audio(audio_file)
                if text:
                    print(f"\n인식된 텍스트: {text}")
                    response = self.get_ai_response(text)
                    await self.speak_async(response)
                    await asyncio.sleep(0.2)

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

    def run(self):
        """메인 실행 함수"""
        asyncio.run(self.run_async())

    def __del__(self):
        """클래스 소멸자"""
        if hasattr(self, 'executor'):
            self.executor.shutdown()
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