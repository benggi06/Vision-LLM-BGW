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

# 노란색 HSV 색상 범위
YELLOW_LOWER = np.array([15, 70, 70], dtype=np.uint8)
YELLOW_UPPER = np.array([36, 255, 255], dtype=np.uint8)

# 자동 검사 트리거 기준값
AUTO_STABILITY_FRAMES = 20  # 약 0.7~1초간 안정 시 자동 검사
STABILITY_DIST_THRESH = 15   # BBox 중심 이동 허용치 (px)
AUTO_COOLDOWN_FRAMES = 45   # 검사 완료 후 재검사 방지 대기 프레임

# ==================== 2. TTS 음성 출력 함수 ====================
def speak_via_piper_venv(text: str):
    """Piper TTS 음성 합성 및 aplay 재생 (오류 시 기본 사운드 장치 자동 우회)"""
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

        for dev in [PRIMARY_SPEAKER, "default", None]:
            cmd = ["aplay"]
            if dev is not None:
                cmd += ["-D", dev]
            cmd.append(AUDIO_OUTPUT)

            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                break
    except Exception as e:
        print(f"[TTS 에러] {e}")

def notify(console_msg: str, speech_msg: str = None):
    print(console_msg)
    speak_text = speech_msg if speech_msg is not None else console_msg
    speak_via_piper_venv(speak_text)

# ==================== 3. 기하학 분석 & 결함 부위 검출 ====================
def analyze_yellow_paper_defects(frame):
    """
    노란색 세그멘테이션, 외곽선 분석 및 찢어진 결함 포인트(Convexity Defect) 추출
    """
    h, w = frame.shape[:2]
    blr = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    margin_x, margin_y = int(w * 0.15), int(h * 0.12)
    default_bbox = (margin_x, margin_y, w - margin_x, h - margin_y)

    if not contours:
        return default_bbox, False, 0.0, "NO_PAPER", None, []

    largest_cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_cnt)

    if area < (h * w * 0.03):
        return default_bbox, False, 0.0, "TOO_SMALL", None, []

    x, y, bw, bh = cv2.boundingRect(largest_cnt)
    x1, y1 = max(0, x - 10), max(0, y - 10)
    x2, y2 = min(w, x + bw + 10), min(h, y + bh + 10)
    bbox = (x1, y1, x2, y2)

    # 1) 꼭짓점 수 & 다각형 근사
    peri = cv2.arcLength(largest_cnt, True)
    approx = cv2.approxPolyDP(largest_cnt, 0.035 * peri, True)
    vertices_count = len(approx)

    # 2) 면적 일치율
    rect = cv2.minAreaRect(largest_cnt)
    rect_area = rect[1][0] * rect[1][1]
    rect_ratio = (area / rect_area) if rect_area > 0 else 0.0

    # 3) 볼록성 및 찢김 결함 위치(Defect Points) 추출
    is_convex = cv2.isContourConvex(approx)
    defect_points = []

    hull_indices = cv2.convexHull(largest_cnt, returnPoints=False)
    if len(hull_indices) > 3 and len(largest_cnt) > 3:
        try:
            defects = cv2.convexityDefects(largest_cnt, hull_indices)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    # 일정 깊이(15px) 이상 파인 찢김/파손 틈새를 결함 포인트로 수집
                    if (d / 256.0) > 15.0:
                        far_pt = tuple(largest_cnt[f][0])
                        defect_points.append(far_pt)
        except Exception:
            pass

    is_valid_rect = (vertices_count == 4) and (rect_ratio >= 0.86) and is_convex and (len(defect_points) == 0)
    detail_str = f"V:{vertices_count} | Ratio:{rect_ratio:.2f} | Defects:{len(defect_points)}"

    return bbox, is_valid_rect, rect_ratio, detail_str, largest_cnt, defect_points

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

# ==================== 5. 메인 실행 루프 (OSD & 자동화) ====================
cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit(1)

cv2.namedWindow("Smart Paper Quality Inspector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smart Paper Quality Inspector", 960, 540)

notify(
    "\n=======================================================\n"
    "🔔 [시스템 시작] 지능형 종이 품질 자동 검사 시스템을 시작합니다.\n"
    "=======================================================",
    "지능형 종이 품질 검사 시스템을 시작합니다."
)

notify(
    "👉 [안내] 종이를 화면에 1초간 가만히 비추면 자동 검사합니다. (수동 검사는 SPACE, 종료는 Q)",
    "종이를 화면에 맞추고 잠시 멈추면 자동으로 검사합니다."
)

# OSD 통계 및 상태 관리 변수
stats = {
    "total": 0,
    "pass": 0,
    "fail": 0,
    "defects": {"TORN": 0, "WRINKLED": 0, "STAINED": 0, "NON_PAPER": 0}
}

status_label = "READY (Hold still for Auto-Inspection)"
box_color = (255, 255, 0)
last_defect_points = []

# 모션 안정성 추적 변수
prev_center = (0, 0)
stable_frame_count = 0
cooldown_timer = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        display_frame = frame.copy()
        h, w = frame.shape[:2]

        if cooldown_timer > 0:
            cooldown_timer -= 1

        # 1) 형상 및 에지 분석
        bbox, is_geom_valid, rect_score, detail_str, cnt, defects = analyze_yellow_paper_defects(frame)
        x1, y1, x2, y2 = bbox
        cur_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # 2) 움직임 안정성 계산 (Auto-Trigger)
        dist_moved = np.sqrt((cur_center[0] - prev_center[0])**2 + (cur_center[1] - prev_center[1])**2)
        prev_center = cur_center

        if (cnt is not None) and (dist_moved < STABILITY_DIST_THRESH) and (cooldown_timer == 0):
            stable_frame_count += 1
        else:
            stable_frame_count = 0

        stability_pct = min(1.0, stable_frame_count / AUTO_STABILITY_FRAMES)

        # 3) 화면 OSD 대시보드 렌더링 (English Only)
        # 상단 통계 바 (Header Dashboard)
        cv2.rectangle(display_frame, (0, 0), (w, 50), (30, 30, 30), -1)
        yield_rate = (stats["pass"] / stats["total"] * 100.0) if stats["total"] > 0 else 0.0
        stat_text = f"TOTAL: {stats['total']}  |  PASS: {stats['pass']}  |  FAIL: {stats['fail']}  |  YIELD: {yield_rate:.1f}%"
        cv2.putText(display_frame, stat_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # 결함 통계 서브 텍스트
        defect_text = f"DEFECTS -> TORN: {stats['defects']['TORN']} | WRINKLE: {stats['defects']['WRINKLED']} | STAIN: {stats['defects']['STAINED']} | NON-PAPER: {stats['defects']['NON_PAPER']}"
        cv2.putText(display_frame, defect_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # 외곽 에지 표시
        if cnt is not None:
            cv2.drawContours(display_frame, [cnt], -1, (0, 255, 255), 2)

        # 결함 지점(Torn spots) 빨간색 원으로 마킹
        for pt in defects:
            cv2.circle(display_frame, pt, 8, (0, 0, 255), -1)
            cv2.circle(display_frame, pt, 14, (0, 0, 255), 2)

        # Bounding Box 및 상태 라벨
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)
        label_text = f"[{status_label}]"
        (txt_w, txt_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(display_frame, (x1, max(0, y1 - txt_h - 10)), (x1 + txt_w + 10, y1), box_color, -1)
        cv2.putText(display_frame, label_text, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

        # 하단 조작 가이드 & 자동 검사 진행 게이지
        cv2.putText(display_frame, f"Geometry: {detail_str}", (20, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Stability Progress Bar
        bar_x, bar_y, bar_w, bar_h = 20, h - 25, 250, 14
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + int(bar_w * stability_pct), bar_y + bar_h), (0, 255, 0), -1)
        cv2.putText(display_frame, f"Auto-Inspect: {int(stability_pct*100)}%", (bar_x + bar_w + 12, bar_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Smart Paper Quality Inspector", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord('q'), ord('Q'), 27]:
            notify(
                f"\n🛑 [시스템 종료] 최종 수율 {yield_rate:.1f}%로 검사를 종료합니다.",
                "시스템을 종료합니다. 수고하셨습니다."
            )
            break

        # 4) 검사 조건 충족 (자동 안정화 트리거 OR 수동 키 입력)
        trigger_inspect = (stable_frame_count >= AUTO_STABILITY_FRAMES) or (key in [32, 13])

        if trigger_inspect:
            stable_frame_count = 0
            cooldown_timer = AUTO_COOLDOWN_FRAMES  # 연속 중복 검사 차단

            status_label = "ANALYZING..."
            box_color = (0, 165, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)
            cv2.imshow("Smart Paper Quality Inspector", display_frame)
            cv2.waitKey(1)

            notify("\n🔍 [검사 진행] 형상 및 결함 요인을 정밀 분석 중입니다...", "종이 상태를 분석 중입니다.")
            stats["total"] += 1

            # 1차 기하학 필터링 (종이 미감지 또는 극단적 찢김)
            if detail_str.startswith("NO_PAPER") or detail_str.startswith("TOO_SMALL"):
                stats["fail"] += 1
                stats["defects"]["NON_PAPER"] += 1
                status_label = "FAIL - NO PAPER DETECTED"
                box_color = (0, 0, 255)
                notify("📄 [판정 결과] 불량 (종이가 감지되지 않음)", "종이가 감지되지 않았거나 너무 작습니다.")
                continue

            if not is_geom_valid:
                stats["fail"] += 1
                stats["defects"]["TORN"] += 1
                status_label = "FAIL - GEOMETRIC DEFECT / TORN"
                box_color = (0, 0, 255)
                notify(f"📄 [판정 결과] 불량 (외곽선 찢김 및 비정형 결함: {detail_str})", "검사 결과, 종이가 찢어졌거나 직사각형이 아닙니다.")
                continue

            # 2차 VLM 다중 결함 정밀 분류 (PASS, TORN, WRINKLED, STAINED, NON_PAPER)
            paper_crop = frame[y1:y2, x1:x2]
            if paper_crop.size == 0:
                paper_crop = frame

            _, buffer = cv2.imencode('.jpg', paper_crop)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            image_data = "data:image/jpeg;base64," + image_base64

            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict industrial paper defect classification expert.\n"
                            "Inspect the cropped yellow paper image and classify it into EXACTLY ONE of the following categories:\n\n"
                            "1. PASS : Clean, completely flat, undamaged, perfect yellow rectangular paper sheet.\n"
                            "2. TORN : Ripped, torn, notched, missing corners, or jagged edges.\n"
                            "3. WRINKLED : Heavily crumpled, wrinkled, folded, or warped surface.\n"
                            "4. STAINED : Dirty, smudged, scribbled, ink-marked, or stained surface.\n"
                            "5. NON_PAPER : Hands, desk surface, phone, or any object that is not a paper sheet.\n\n"
                            "Constraint:\n"
                            "Output ONLY ONE word from: [PASS, TORN, WRINKLED, STAINED, NON_PAPER]."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Classify the quality of this yellow paper sheet into [PASS, TORN, WRINKLED, STAINED, NON_PAPER]."
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

            decision = response["choices"][0]["message"]["content"].strip().upper()

            # 5) 판정 결과 처리 및 맞춤형 음성 피드백
            if "PASS" in decision:
                stats["pass"] += 1
                status_label = "PASS - CLEAN & INTACT"
                box_color = (0, 255, 0)
                console_res = "📄 [판정 결과] 정상 (PASS - 온전한 노란색 직사각형 종이)"
                voice_res = "검사 결과, 온전한 정상 규격 종이입니다. 사용 가능합니다."
            elif "WRINKLED" in decision:
                stats["fail"] += 1
                stats["defects"]["WRINKLED"] += 1
                status_label = "FAIL - CRUMPLED / WRINKLED"
                box_color = (0, 0, 255)
                console_res = "📄 [판정 결과] 불량 (구김 결함 - WRINKLED)"
                voice_res = "검사 결과, 종이가 심하게 구겨져 있어 불량입니다."
            elif "STAINED" in decision:
                stats["fail"] += 1
                stats["defects"]["STAINED"] += 1
                status_label = "FAIL - CONTAMINATED / STAINED"
                box_color = (0, 0, 255)
                console_res = "📄 [판정 결과] 불량 (오염/낙서 결함 - STAINED)"
                voice_res = "검사 결과, 종이 표면에 오염이나 얼룩이 있어 불량입니다."
            elif "NON_PAPER" in decision:
                stats["fail"] += 1
                stats["defects"]["NON_PAPER"] += 1
                status_label = "FAIL - NOT A PAPER"
                box_color = (0, 0, 255)
                console_res = "📄 [판정 결과] 불량 (종이가 아닌 물체 감지 - NON_PAPER)"
                voice_res = "종이가 감지되지 않았거나 올바른 종이가 아닙니다. 다시 비춰주세요."
            else:  # TORN 또는 기타 결함
                stats["fail"] += 1
                stats["defects"]["TORN"] += 1
                status_label = "FAIL - TORN / SURFACE DAMAGE"
                box_color = (0, 0, 255)
                console_res = "📄 [판정 결과] 불량 (찢어짐/파손 결함 - TORN)"
                voice_res = "검사 결과, 종이에 찢어짐이나 손상이 발견되어 불량입니다."

            notify(console_res, voice_res)

finally:
    cap.release()
    cv2.destroyAllWindows()

    if 'llm' in locals():
        del llm
    if 'chat_handler' in locals():
        del chat_handler
    gc.collect()
