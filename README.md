import os
import subprocess
import time
import base64
import gc
import cv2
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

# ==================== 1. 경로 및 장치 설정 ====================
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"

PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
AUDIO_DIR = "src/audio"
AUDIO_OUTPUT = os.path.join(AUDIO_DIR, "response.wav")
PRIMARY_SPEAKER = "plughw:2,0"  # 기본 지정 장치

os.makedirs(AUDIO_DIR, exist_ok=True)

# CSI 카메라 GStreamer 파이프라인
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

# ==================== 2. 음성 출력 함수 (안전 폴백 적용) ====================
def speak_via_piper_venv(text: str):
    """
    Piper TTS로 음성 합성 후 aplay로 출력 (장치 오류 시 default로 폴백)
    """
    try:
        # 1) Piper TTS 음성 파일 생성 (.piper_venv)
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

        # 2) 오디오 재생 시도 (PRIMARY_SPEAKER -> default -> 기본 aplay 순차 시도)
        played = False
        for dev in [PRIMARY_SPEAKER, "default", None]:
            cmd = ["aplay"]
            if dev is not None:
                cmd += ["-D", dev]
            cmd.append(AUDIO_OUTPUT)

            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                played = True
                break

        if not played:
            print("[오디오 경고] 스피커 재생 실패. 'aplay -l' 명령어로 오디오 장치 번호를 확인하세요.")

    except Exception as e:
        print(f"[TTS 에러] {e}")

def notify(console_msg: str, speech_msg: str = None):
    print(console_msg)
    speak_text = speech_msg if speech_msg is not None else console_msg
    speak_via_piper_venv(speak_text)

# ==================== 3. VLM 모델 로드 ====================
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

# ==================== 4. 메인 실행 루프 (카메라 GUI 창 포함) ====================
cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다. 파이프라인 설정을 확인하세요.")
    exit(1)

cv2.namedWindow("Paper Inspection Camera View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Paper Inspection Camera View", 960, 540)

# 시작 안내
notify(
    "\n========================================\n"
    "🔔 [시스템 시작] 종이 상태 검사 시스템을 시작합니다.\n"
    "========================================",
    "종이 상태 검사 시스템을 시작합니다."
)

notify(
    "👉 [안내] 화면에 종이를 맞추고 [SPACE] 또는 [ENTER] 키를 누르면 검사합니다. (종료는 Q)",
    "화면을 보며 종이를 맞추고 스페이스바를 눌러주세요."
)

last_result_text = "Ready (Press SPACE to Inspect)"
last_color = (255, 255, 0)  # Cyan

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임 수신 대기 중...")
            time.sleep(0.1)
            continue

        # 화면 출력용 프레임 복사
        display_frame = frame.copy()

        # 화면 하단에 조작 가이드 오버레이
        cv2.putText(display_frame, "Key: [SPACE/ENTER] Inspect | [Q] Quit", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 이전 검사 결과 상태 표시
        cv2.putText(display_frame, f"Status: {last_result_text}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2, cv2.LINE_AA)

        cv2.imshow("Paper Inspection Camera View", display_frame)

        # 키 입력 대기 (1ms)
        key = cv2.waitKey(1) & 0xFF

        # 종료 (Q 또는 ESC)
        if key == ord('q') or key == ord('Q') or key == 27:
            notify(
                "\n🛑 [시스템 종료] 종이 상태 검사 시스템을 종료합니다.",
                "시스템을 종료합니다. 이용해 주셔서 감사합니다."
            )
            break

        # 검사 실행 (스페이스바: 32, 엔터키: 13)
        if key == 32 or key == 13:
            # 1) 화면에 분석 중 표시 업데이트
            analyzing_frame = frame.copy()
            cv2.putText(analyzing_frame, "Analyzing Paper Quality...", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.imshow("Paper Inspection Camera View", analyzing_frame)
            cv2.waitKey(1)

            # 2) 검사 시작 음성 송출
            notify(
                "\n🔍 [검사 진행] 종이 상태를 분석 중입니다...",
                "종이 상태를 분석 중입니다."
            )

            # 3) 메모리 상에서 Base64 인코딩
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            image_data = "data:image/jpeg;base64," + image_base64

            # 4) Gemma 4 VLM 판단
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Instruction:\n"
                            "주어진 이미지 속 종이의 찢김 및 훼손 여부를 확인하여 사용 가능 여부를 판단하시오.\n\n"
                            "Constraint:\n"
                            "종이가 온전하고 찢어지지 않았다면 반드시 '사용 가능', 찢어지거나 구멍이 났다면 반드시 '사용 불가'라고만 답하시오.\n"
                            "다른 설명은 절대 붙이지 마시오."
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

            # 5) 판정 결과 처리 및 음성 안내
            if "사용 가능" in decision:
                console_res = "📄 [판정 결과] 사용 가능 (상태: 온전함)"
                voice_res = "검사 결과, 종이가 온전하여 사용 가능합니다."
                last_result_text = "PASS (Usable)"
                last_color = (0, 255, 0)  # Green
            else:
                console_res = "📄 [판정 결과] 사용 불가 (상태: 찢어짐/훼손)"
                voice_res = "검사 결과, 종이가 찢어져 있어 사용 불가능합니다."
                last_result_text = "FAIL (Torn / Defective)"
                last_color = (0, 0, 255)  # Red

            notify(console_res, voice_res)

finally:
    # 안전 리소스 해제 루틴
    cap.release()
    cv2.destroyAllWindows()
    
    # Llama 객체 사전 정리로 종료 예외 차단
    if 'llm' in locals():
        del llm
    if 'chat_handler' in locals():
        del chat_handler
    gc.collect()
