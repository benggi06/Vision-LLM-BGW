# Vision-LLM-260824
[금오공대] Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템

import os
import subprocess
import time
import base64
import cv2
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

# ==================== 1. 경로 및 장치 설정 ====================
# .venv 환경에서 사용하는 VLM 모델 경로
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"

# .piper_venv 환경의 파이썬 인터프리터 및 TTS 설정
PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
AUDIO_DIR = "src/audio"
AUDIO_OUTPUT = os.path.join(AUDIO_DIR, "response.wav")
SPEAKER_DEVICE = "plughw:2,0"

# 오디오 저장 디렉토리 생성
os.makedirs(AUDIO_DIR, exist_ok=True)

# Jetson CSI 카메라 파이프라인
PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# ==================== 2. 텍스트 + 음성 동시 출력 함수 ====================
def speak_via_piper_venv(text: str):
    """
    .piper_venv 환경의 파이썬을 subprocess로 호출해 TTS 음성 생성 및 aplay 재생
    """
    try:
        # 1) Piper TTS 음성 합성 (.piper_venv)
        subprocess.run(
            [
                PIPER_PYTHON,
                "-m", "piper",
                "-m", PIPER_MODEL,
                "-f", AUDIO_OUTPUT,
                "--", text,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # 2) 스피커로 aplay 재생 (시스템 레벨)
        subprocess.run(
            ["aplay", "-D", SPEAKER_DEVICE, AUDIO_OUTPUT],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"[TTS 에러] 음성 출력 중 문제가 발생했습니다: {e}")

def notify(console_msg: str, speech_msg: str = None):
    """
    쉘 콘솔에 텍스트를 출력하고, 스피커로 음성을 동시에 송출하는 공통 함수
    """
    print(console_msg)
    speak_text = speech_msg if speech_msg is not None else console_msg
    speak_via_piper_venv(speak_text)

# ==================== 3. VLM 모델 로드 (.venv) ====================
print("\n[시스템 준비] VLM 모델을 로드하는 중입니다. 잠시만 기다려주세요...")
chat_handler = Gemma4ChatHandler(clip_model_path=MMPROJ_PATH)
llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    chat_handler=chat_handler,
    n_gpu_layers=-1,
    n_ctx=2048,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)

# ==================== 4. 메인 실행 루프 ====================
cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)

# 시작 안내 음성 및 텍스트 출력
notify(
    "\n========================================\n"
    "🔔 [시스템 시작] 종이 상태 검사 시스템을 시작합니다.\n"
    "========================================",
    "종이 상태 검사 시스템을 시작합니다."
)

notify(
    "👉 [안내] 종이를 카메라에 비추고 엔터를 누르세요. (종료는 q 입력 후 엔터)",
    "종이를 카메라에 비추고 엔터를 눌러주세요."
)

try:
    while True:
        user_cmd = input("\n[대기 중] Enter: 검사 시작 | q: 종료 > ")
        
        # 종료 명령 처리
        if user_cmd.strip().lower() in ["q", "exit", "quit", "종료"]:
            notify(
                "\n🛑 [시스템 종료] 종이 상태 검사 시스템을 종료합니다.",
                "시스템을 종료합니다. 이용해 주셔서 감사합니다."
            )
            break

        # 1) 검사 시작 안내 (콘솔 + 음성)
        notify(
            "\n🔍 [검사 진행] 종이 상태 검사를 시작합니다. 분석 중입니다...",
            "종이 상태 검사를 시작합니다. 잠시만 기다려주세요."
        )

        # 2) 카메라 프레임 수집
        ret, frame = cap.read()
        if not ret:
            notify(
                "❌ [오류] 카메라 프레임을 읽어오지 못했습니다. 다시 시도해주세요.",
                "카메라 영상을 읽어오지 못했습니다. 다시 시도해주세요."
            )
            continue

        # 3) 파일 생성 없이 메모리에서 바로 Base64 변환
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        image_data = "data:image/jpeg;base64," + image_base64

        # 4) Gemma 4 VLM 판단 (.venv)
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Instruction:\n"
                        "주어진 이미지 속 종이의 찢김 및 훼손 여부를 확인하여 사용 가능 여부를 판단하시오.\n\n"
                        "Constraint:\n"
                        "종이가 온전하고 찢어지지 않았다면 반드시 '사용 가능', 찢어지거나 구멍이 났다면 반드시 '사용 불가'라고만 답하시오.\n"
                        "다른 단어나 문장은 절대 추가하지 마시오."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "이 종이의 찢어짐 여부와 사용 가능 여부를 판정해주세요."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data}
                        },
                    ]
                }
            ],
            max_tokens=15,
            temperature=0.0,
        )

        decision = response["choices"][0]["message"]["content"].strip()

        # 5) 판별 결과에 따른 콘솔 및 음성 메시지 분기 송출
        if "사용 가능" in decision:
            console_res = f"📄 [판정 결과] 사용 가능 (상태: 온전함)"
            voice_res = "검사 결과, 종이가 온전하여 사용 가능합니다."
        else:
            console_res = f"📄 [판정 결과] 사용 불가 (상태: 찢어짐/훼손)"
            voice_res = "검사 결과, 종이가 찢어져 있어 사용 불가능합니다."

        notify(console_res, voice_res)

finally:
    cap.release()
    cv2.destroyAllWindows()
