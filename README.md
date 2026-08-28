import os
import subprocess
import time
import base64
import gc
import cv2
import numpy as np
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

# ==================== 1. 경로 및 하드웨어 설정 ====================
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"

PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
AUDIO_DIR = "src/audio"
AUDIO_OUTPUT = os.path.join(AUDIO_DIR, "response.wav")
PRIMARY_SPEAKER = "plughw:2,0"

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

# ==================== 2. TTS 음성 출력 함수 ====================
def speak_via_piper_venv(text: str):
    """Piper TTS로 음성 합성 후 aplay로 출력 (장치 오류 시 기본 장치로 자동 폴백)"""
    try:
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
            print("[오디오 경고] 스피커 출력을 확인하세요 (aplay -l).")

    except Exception as e:
        print(f"[TTS 에러] {e}")

def notify(console_msg: str, speech_msg: str = None):
    print(console_msg)
    speak_text = speech_msg if speech_msg is not None else console_msg
    speak_via_piper_venv(speak_text)

# ==================== 3. 종이 Bounding Box 검출 함수 ====================
def find_paper_bounding_box(frame):
    """
    영상 처리 기반으로 화면 내 가장 유력한 종이 객체의 Bounding Box(x1, y1, x2, y2)를 계산
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 에지 검출 및 모폴로지 클로징 연산
    edges = cv2.Canny(blurred, 30, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_box = None
    if len(contours) > 0:
        # 화면의 일정 크기 이상인 가장 큰 컨투어 탐색
        largest_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        if area > (h * w * 0.04):
            x, y, bw, bh = cv2.boundingRect(largest_cnt)
            # 여유 마진 추가
            x1 = max(0, x - 15)
            y1 = max(0, y - 15)
            x2 = min(w, x + bw + 15)
            y2 = min(h, y + bh + 15)
            valid_box = (x1, y1, x2, y2)

    # 명확한 객체가 잡히지 않을 경우 기본 중심 영역 설정
    if valid_box is None:
        margin_x, margin_y = int(w * 0.15), int(h * 0.12)
        valid_box = (margin_x, margin_y, w - margin_x, h - margin_y)

    return valid_box

# ==================== 4. VLM 모델 로드 ====================
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

# ==================== 5. 메인 실행 루프 ====================
cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit(1)

cv2.namedWindow("Paper Quality Inspector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Paper Quality Inspector", 960, 540)

# 시작 안내
notify(
    "\n========================================\n"
    "🔔 [시스템 시작] 종이 상태 검사 시스템을 시작합니다.\n"
    "========================================",
    "종이 상태 검사 시스템을 시작합니다."
)

notify(
    "👉 [안내] 화면 박스에 종이를 맞추고 [SPACE] 또는 [ENTER]를 누르세요. (종료는 Q)",
    "화면을 보며 종이를 맞추고 스페이스바를 눌러주세요."
)

# 상태 저장 변수 (화면 표시는 영문 전용)
status_label = "READY (Press SPACE to Inspect)"
box_color = (255, 255, 0)  # 초기 대기: 하늘색

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        display_frame = frame.copy()
        
        # 1) 실시간 Bounding Box 계산
        x1, y1, x2, y2 = find_paper_bounding_box(frame)

        # 2) BBox 및 라벨 표시 (영문 전용)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)

        label_text = f"[{status_label}]"
        (txt_w, txt_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(display_frame, (x1, max(0, y1 - txt_h - 10)), (x1 + txt_w + 10, y1), box_color, -1)
        cv2.putText(display_frame, label_text, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

        # 상단 조작 가이드 (영문)
        cv2.putText(display_frame, "Key: [SPACE / ENTER] Inspect  |  [Q] Quit", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Paper Quality Inspector", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # 종료 (Q / ESC)
        if key in [ord('q'), ord('Q'), 27]:
            notify(
                "\n🛑 [시스템 종료] 종이 검사 시스템을 종료합니다.",
                "시스템을 종료합니다. 이용해 주셔서 감사합니다."
            )
            break

        # 검사 실행 (SPACE: 32 / ENTER: 13)
        if key in [32, 13]:
            # 화면 상태 업데이트
            status_label = "ANALYZING..."
            box_color = (0, 165, 255)  # 주황색
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)
            cv2.imshow("Paper Quality Inspector", display_frame)
            cv2.waitKey(1)

            notify("\n🔍 [검사 진행] 종이 형태 및 찢어짐을 분석 중입니다...", "종이 상태를 분석 중입니다.")

            # BBox 영역 크롭
            paper_crop = frame[y1:y2, x1:x2]
            if paper_crop.size == 0:
                paper_crop = frame

            # 메모리 상에서 Base64 인코딩
            _, buffer = cv2.imencode('.jpg', paper_crop)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            image_data = "data:image/jpeg;base64," + image_base64

            # Gemma 4 VLM 정밀 판단 (비종이 객체 차단 + 직사각형 및 찢김 검증)
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert industrial quality control inspector for paper sheets.\n\n"
                            "Strict Inspection Rules:\n"
                            "1. Object Verification: Check if the main object is a genuine sheet of paper. If it is NOT paper (e.g. human hand, face, desk surface, phone, mug, keyboard, pen, background only, or other items), you MUST output: 불량\n"
                            "2. Defect Detection: If it is paper, check for any tears, rips, jagged edges, cracks, missing corners, holes, or non-rectangular shapes. If any defect is found, you MUST output: 불량\n"
                            "3. Pass Criteria: ONLY if it is a real, completely intact, clean, flat rectangular sheet of paper with no tears, output: 정상\n\n"
                            "Constraint:\n"
                            "Output EXACTLY ONE word: '정상' or '불량'. Do not write any other explanation or symbol."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Is this a real, clean, undamaged rectangular sheet of paper? Answer 정상 or 불량."
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data}
                            },
                        ]
                    }
                ],
                max_tokens=10,
                temperature=0.0,
            )

            decision = response["choices"][0]["message"]["content"].strip()

            # 판정 결과 분기 (화면: 영어 / 콘솔 및 음성: 한국어)
            if "정상" in decision:
                status_label = "PASS - INTACT RECTANGULAR PAPER"
                box_color = (0, 255, 0)  # 초록색 박스
                console_res = "📄 [판정 결과] 정상 (온전한 직사각형 종이 - 사용 가능)"
                voice_res = "검사 결과, 온전한 직사각형 종이로 정상입니다."
            else:
                status_label = "FAIL - DEFECTIVE / NOT PAPER"
                box_color = (0, 0, 255)  # 빨간색 박스
                console_res = "📄 [판정 결과] 불량 (찢어짐, 비정형 형태 또는 종이가 아님 - 사용 불가)"
                voice_res = "검사 결과, 종이가 찢어졌거나 올바른 종이가 아닙니다. 불량입니다."

            notify(console_res, voice_res)

finally:
    cap.release()
    cv2.destroyAllWindows()

    if 'llm' in locals():
        del llm
    if 'chat_handler' in locals():
        del chat_handler
    gc.collect()
