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
    """Piper TTS로 음성 합성 후 aplay로 출력 (오류 시 기본 장치로 자동 폴백)"""
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
    영상 처리 기반으로 화면 내 가장 큰 종이 객체의 Bounding Box(x1, y1, x2, y2)를 계산
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Otsu 이진화 및 Canny 에지 검출 결합
    edges = cv2.Canny(blurred, 40, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_box = None
    if len(contours) > 0:
        # 화면의 일정 크기 이상인 가장 큰 컨투어 선택
        largest_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        if area > (h * w * 0.05):  # 화면 5% 이상 면적
            x, y, bw, bh = cv2.boundingRect(largest_cnt)
            # 마진 추가
            x1 = max(0, x - 10)
            y1 = max(0, y - 10)
            x2 = min(w, x + bw + 10)
            y2 = min(h, y + bh + 10)
            valid_box = (x1, y1, x2, y2)

    # 종이 영역이 명확히 안 잡히면 화면 중앙 기본 영역 지정
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
    "🔔 [시스템 시작] 종이 외관 검사 시스템을 시작합니다.\n"
    "========================================",
    "종이 외관 검사 시스템을 시작합니다."
)

notify(
    "👉 [안내] 화면 박스에 종이를 맞추고 [SPACE]를 누르세요. (종료는 Q)",
    "화면을 보며 종이를 맞추고 스페이스바를 눌러주세요."
)

# 상태 저장 변수
current_status = "READY"
status_label = "READY (Press SPACE to Inspect)"
box_color = (255, 255, 0)  # 초기 대기: 하늘색

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        display_frame = frame.copy()
        
        # 1) 실시간 종이 위치 Bounding Box 계산
        x1, y1, x2, y2 = find_paper_bounding_box(frame)

        # 2) 화면 상에 Object Detection 형태의 BBox 및 라벨 표시
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)

        # 라벨 배경 및 텍스트 표시
        label_text = f"[{status_label}]"
        (txt_w, txt_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(display_frame, (x1, max(0, y1 - txt_h - 12)), (x1 + txt_w + 10, y1), box_color, -1)
        cv2.putText(display_frame, label_text, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # 상단 안내 바
        cv2.putText(display_frame, "Key: [SPACE/ENTER] Inspect | [Q] Quit", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Paper Quality Inspector", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # 종료 (Q / ESC)
        if key in [ord('q'), ord('Q'), 27]:
            notify(
                "\n🛑 [시스템 종료] 종이 검사 시스템을 종료합니다.",
                "시스템을 종료합니다. 이용해 주셔서 감사합니다."
            )
            break

        # 검사 트리거 (SPACE: 32 / ENTER: 13)
        if key in [32, 13]:
            # 화면에 분석 중 알림 갱신
            status_label = "ANALYZING..."
            box_color = (0, 165, 255)  # 주황색
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)
            cv2.imshow("Paper Quality Inspector", display_frame)
            cv2.waitKey(1)

            notify("\n🔍 [검사 진행] 종이 형태 및 찢어짐을 분석 중입니다...", "종이 상태를 분석 중입니다.")

            # BBox 영역을 크롭하여 판별 정확도 상승
            paper_crop = frame[y1:y2, x1:x2]
            if paper_crop.size == 0:
                paper_crop = frame

            # 메모리 상에서 Base64 인코딩
            _, buffer = cv2.imencode('.jpg', paper_crop)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            image_data = "data:image/jpeg;base64," + image_base64

            # Gemma 4 VLM 판단 (직사각형 여부 + 찢어짐/구멍 훼손 정밀 검증)
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Instruction:\n"
                            "Inspect the paper in the image carefully.\n"
                            "Determine if the paper is a clean, fully intact rectangular shape or defective.\n\n"
                            "Defect Criteria (불량):\n"
                            "- The paper is torn, ripped, or has cut marks.\n"
                            "- The edges are jagged, irregular, or missing corners.\n"
                            "- There are visible cracks, gaps, holes, or non-rectangular shapes.\n\n"
                            "Pass Criteria (정상):\n"
                            "- The paper is a complete, clean, smooth rectangle without any tears or missing parts.\n\n"
                            "Constraint:\n"
                            "Respond with ONLY ONE word: '정상' or '불량'."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "종이가 찢어지지 않은 온전한 직사각형인지 검사하여 판정하시오."
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

            # 판정 결과 분기 및 상태 업데이트
            if "정상" in decision:
                current_status = "PASS"
                status_label = "PASS (정상 - 사용 가능)"
                box_color = (0, 255, 0)  # 초록색 박스
                console_res = "📄 [판정 결과] 정상 (직사각형 온전함 / 사용 가능)"
                voice_res = "검사 결과, 종이가 온전한 직사각형으로 정상입니다."
            else:
                current_status = "FAIL"
                status_label = "FAIL (찢어짐/불량 - 사용 불가)"
                box_color = (0, 0, 255)  # 빨간색 박스
                console_res = "📄 [판정 결과] 불량 (찢어짐 또는 불규칙 형태 / 사용 불가)"
                voice_res = "검사 결과, 종이가 찢어졌거나 형태가 불량하여 사용 불가합니다."

            notify(console_res, voice_res)

finally:
    cap.release()
    cv2.destroyAllWindows()

    # 프로세스 종료 시 메모리/바인딩 안전 해제
    if 'llm' in locals():
        del llm
    if 'chat_handler' in locals():
        del chat_handler
    gc.collect()
