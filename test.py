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
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from fuzzywuzzy import fuzz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 환경 변수 설정
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# 경고 메시지 필터링
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
location_data = pd.read_excel("./data/기상청_단기예보_위경도.xlsx")

def convert_coordinates(row):
    """시분초 형식의 좌표를 십진수 도(degree) 형식으로 변환"""
    longitude = float(row['경도(시)']) + float(row['경도(분)'])/60 + float(row['경도(초)'])/3600
    latitude = float(row['위도(시)']) + float(row['위도(분)'])/60 + float(row['위도(초)'])/3600
    return longitude, latitude


def find_coordinates(location_name):
    best_match = None
    best_score = 0
    latitude = None
    longitude = None

    for idx, row in location_data.iterrows():
        # 단계별 유사도 비교
        score_1 = fuzz.ratio(location_name, row['1단계']) if pd.notnull(row['1단계']) else 0
        score_2 = fuzz.ratio(location_name, row['2단계']) if pd.notnull(row['2단계']) else 0
        score_3 = fuzz.ratio(location_name, row['3단계']) if pd.notnull(row['3단계']) else 0
        
        # 가중치를 적용한 최종 유사도 계산
        total_score = (score_1 * 0.5) + (score_2 * 0.3) + (score_3 * 0.2)

        # 가장 높은 유사도 점수를 가진 지역 선택
        if total_score > best_score:
            best_score = total_score
            best_match = f"{row['1단계']} {row['2단계']} {str(row['3단계'])}"
            latitude = row['위도']
            longitude = row['경도']

    return {
        '입력 지명': location_name,
        '일치 지명': best_match,
        '일치율': best_score,
        '위도': latitude,
        '경도': longitude
    }


class LocationMatcher:
    def __init__(self, location_data):
        self.vectorizer = TfidfVectorizer(
            analyzer='char', 
            ngram_range=(1, 4),
            min_df=2,
            max_df=0.9
        )
        self.location_data = location_data
        self.locations = [
            f"{row['1단계']} {row['2단계']} {str(row['3단계'])}"
            for _, row in location_data.iterrows()
        ]
        self.vectors = self.vectorizer.fit_transform(self.locations)
        
    def find_coordinates(self, location_name):
        query_vec = self.vectorizer.transform([location_name])
        similarities = cosine_similarity(query_vec, self.vectors)[0]
        best_idx = np.argmax(similarities)
        
        row = self.location_data.iloc[best_idx]
        longitude, latitude = convert_coordinates(row)
        
        print(f"\n[DEBUG] 검색 위치: {location_name}")
        print(f"[DEBUG] 매칭 점수: {similarities[best_idx]:.2f}")
        print(f"[DEBUG] 매칭 위치: {self.locations[best_idx]}")
        print(f"[DEBUG] 격자 좌표: X={row['격자 X']}, Y={row['격자 Y']}\n")
        
        return {
            '입력 지명': location_name,
            '일치 지명': self.locations[best_idx],
            '일치율': similarities[best_idx] * 100,
            '위도': latitude,
            '경도': longitude,
            '격자X': row['격자 X'],
            '격자Y': row['격자 Y']
        }


class WeatherFetcher:
    def __init__(self):
        self.base_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
        self.api_key = os.getenv('KMA_API_KEY')
        self.default_x = 55
        self.default_y = 127

    def fetch_weather(self, date=None, time=None, nx=None, ny=None):
        if not date:
            date = datetime.now().strftime("%Y%m%d")
        if not time:
            time = "0500"

        params = {
            'serviceKey': self.api_key,
            'numOfRows': 10,
            'pageNo': 1,
            'dataType': 'JSON',
            'base_date': date,
            'base_time': time,
            'nx': nx if nx else self.default_x,
            'ny': ny if ny else self.default_y
        }
        
        print(f"\n[DEBUG] API 요청 파라미터:")
        print(f"URL: {self.base_url}")
        print(f"Parameters: {params}\n")

        try:
            response = requests.get(self.base_url, params=params)
            print(f"[DEBUG] API 응답 상태: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"[DEBUG] API 응답 데이터: {data}\n")
                return data
            else:
                print(f"[DEBUG] API 오류 응답: {response.text}\n")
                return None
        except Exception as e:
            print(f"[DEBUG] API 요청 실패: {str(e)}\n")
            return None


class WeatherTool:
    def __init__(self):
        self.fetcher = WeatherFetcher()

    def fetch_weather_by_location(self, location_name):
        coordinates = find_coordinates(location_name)  # find_coordinates 함수는 지역명에 대한 좌표를 반환해야 함
        if coordinates['위도'] is not None and coordinates['경도'] is not None:
            weather_data = self.fetcher.fetch_weather(nx=coordinates['경도'], ny=coordinates['위도'])
            return weather_data
        else:
            return None

class ToolManager:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, tool):
        self.tools[name] = tool

    def get_tool_response(self, tool_name, *args, **kwargs):
        if tool_name in self.tools:
            return self.tools[tool_name].fetch_weather_by_location(*args, **kwargs)
        else:
            return "해당 도구를 찾을 수 없습니다."

tool_manager = ToolManager()
tool_manager.register_tool("weather", WeatherTool())

def get_weather_response(location_name):
    print(f"\n[DEBUG] 날씨 조회 시작: {location_name}")
    
    weather_tool = tool_manager.tools["weather"]
    
    matcher = LocationMatcher(location_data)
    find_coordinates = matcher.find_coordinates

    coordinates = find_coordinates(location_name)
    print(f"[DEBUG] 좌표 검색 결과: {coordinates}")
    
    weather_data = weather_tool.fetcher.fetch_weather(
        nx=coordinates['격자X'],
        ny=coordinates['격자Y']
    )
    
    if weather_data:
        items = weather_data['response']['body']['items']['item']
        weather_info = []
        print("\n[DEBUG] 날씨 정보 파싱:")
        location = coordinates["일치 지명"]
        weather_info.append(f"지역: {location}")
        for item in items:
            category = item['category']
            value = item['fcstValue']
            print(f"[DEBUG] 카테고리: {category}, 값: {value}")

            if category == 'TMP':
                weather_info.append(f"기온: {value}°C")
            elif category == 'SKY':
                sky_status = {"1": "맑음", "3": "구름많음", "4": "흐림"}.get(value, "알 수 없음")
                weather_info.append(f"하늘 상태: {sky_status}")
            elif category == 'PTY':
                rain_status = {"0": "없음", "1": "비", "2": "비/눈", "3": "눈"}.get(value, "알 수 없음")
                weather_info.append(f"강수 형태: {rain_status}")
        
        result = f"\n ".join(weather_info)
        print(f"\n[DEBUG] 최종 결과: {result}")
        return result
    else:
        print("\n[DEBUG] 날씨 데이터 없음")
        return "날씨 정보를 가져오는 데 실패했습니다."
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

            # 도구 관리자 초기화
            self.tool_manager = ToolManager()
            self.tool_manager.register_tool("weather", WeatherTool())

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
        """OpenAI API를 사용하여 응답 생성 (대화 히스토리 포함)"""
        try:
            self.message_history.append({"role": "user", "content": text})
            if "날씨" in text:
                location_name = re.search(r"날씨\s*([^ ]+)", text)
                if location_name:
                    location_name = location_name.group(1)
                    weather_response = get_weather_response(location_name)
                    self.message_history.append({"role": "assistant", "content": weather_response})
                    return weather_response
            
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
        return [sent.strip() for sent in text.split('\n') if sent.strip()]

    async def speak_async(self, text):
        """문장 단위로 분할하여 비동기적으로 TTS 처리"""
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
