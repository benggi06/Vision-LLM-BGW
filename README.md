import os
import json
import subprocess
import time
import base64
import gc
from datetime import datetime
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

OUTPUT_DIR = "src/output"
JSON_HISTORY_PATH = os.path.join(OUTPUT_DIR, "inspection_history.json")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# 노란색 HSV 색상 범위 (자연광 및 손 그림자 고려)
YELLOW_LOWER = np.array([15, 60, 60], dtype=np.uint8)
YELLOW_UPPER = np.array([38, 255, 255], dtype=np.uint8)

# 자동 검사 트리거 기준값
AUTO_STABILITY_FRAMES = 20  # 약 0.7초간 정지 시 자동 캡처
STABILITY_DIST_THRESH = 18   # BBox 이동 허용 오차 (px)
AUTO_COOLDOWN_FRAMES = 45   # 검사 후 중복 실행 방지 쿨다운

# ==================== 2. 음성 출력 함수 ====================
def speak_via_piper_venv(text: str):
    """Piper TTS 음성 합성 및 aplay 재생 (오류 시 기본 장치 자동 폴백)"""
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

# ==================== 3. 기하학 분석 (손가락 가림 허용 및 기준 완화) ====================
def analyze_yellow_paper_relaxed(frame):
    """
    노란색 종이 영역 검출 (손잡음 형태 허용, 완화된 직사각형 비율 검사)
    """
    h, w = frame.shape[:2]
    blr = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    margin_x, margin_y = int(w * 0.15), int(h * 0.12)
    default_bbox = (margin_x, margin_y, w - margin_x, h - margin_y)

    if not contours:
        return default_bbox, False, 0.0, "NO_PAPER", None, []

    largest_cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_cnt)

    if area < (h * w * 0.04):
        return default_bbox, False, 0.0, "TOO_SMALL", None, []

    x, y, bw, bh = cv2.boundingRect(largest_cnt)
    x1, y1 = max(0, x - 10), max(0, y - 10)
    x2, y2 = min(w, x + bw + 10), min(h, y + bh + 10)
    bbox = (x1, y1, x2, y2)

    # 다각형 근사 (손가락 접촉으로 인한 모서리 증가 허용: 4~8각)
    peri = cv2.arcLength(largest_cnt, True)
    approx = cv2.approxPolyDP(largest_cnt, 0.03 * peri, True)
    vertices_count = len(approx)

    # 최소 외접 직사각형 대비 면적 일치율 (완화: 70% 이상)
    rect = cv2.minAreaRect(largest_cnt)
    rect_area = rect[1][0] * rect[1][1]
    rect_ratio = (area / rect_area) if rect_area > 0 else 0.0

    # 깊은 찢김(Defect)만 필터링 (손가락 굴곡 제외를 위해 깊이 30px 이상만 탐지)
    defect_points = []
    hull_indices = cv2.convexHull(largest_cnt, returnPoints=False)
    if len(hull_indices) > 3 and len(largest_cnt) > 3:
        try:
            defects = cv2.convexityDefects(largest_cnt, hull_indices)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    if (d / 256.0) > 30.0:  # 깊은 찢김만 결함 포인트로 간주
                        far_pt = tuple(largest_cnt[f][0])
                        defect_points.append(far_pt)
        except Exception:
            pass

    # 완화된 기하학적 기준: 면적 비율 70% 이상, 깊은 결함 1개 이하
    is_valid_geom = (rect_ratio >= 0.70) and (len(defect_points) <= 1) and (4 <= vertices_count <= 8)
    detail_str = f"V:{vertices_count} | Ratio:{rect_ratio:.2f} | Rips:{len(defect_points)}"

    return bbox, is_valid_geom, rect_ratio, detail_str, largest_cnt, defect_points

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

cv2.namedWindow("Smart Paper Quality Inspector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smart Paper Quality Inspector", 960, 540)

session_start_dt = datetime.now()
session_start_str = session_start_dt.strftime("%Y-%m-%d %H:%M:%S")

notify(
    "\n=======================================================\n"
    "🔔 [시스템 시작] 지능형 종이 품질 자동 검사 시스템을 시작합니다.\n"
    "=======================================================",
    "지능형 종이 품질 검사 시스템을 시작합니다."
)

notify(
    "👉 [안내] 노란색 종이를 비추고 잠시 멈추면 자동 검사합니다. (수동: SPACE / 종료: Q)",
    "종이를 화면에 맞추고 잠시 멈추면 자동으로 검사합니다."
)

# 품질 통계 관리 (NON_PAPER는 불량에서 제외)
stats = {
    "total_paper": 0,      # 실제 검사된 종이 수 (PASS + FAIL)
    "pass": 0,
    "fail": 0,
    "non_paper_skipped": 0, # 종이가 아니어서 제외된 횟수
    "defects": {"TORN": 0, "WRINKLED": 0, "STAINED": 0}
}

status_label = "READY (Hold still for Auto-Inspection)"
box_color = (255, 255, 0)
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
        bbox, is_geom_valid, rect_score, detail_str, cnt, defects = analyze_yellow_paper_defects_relaxed(frame)
        x1, y1, x2, y2 = bbox
        cur_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # 2) 모션 안정성 추적 (Auto-Trigger)
        dist_moved = np.sqrt((cur_center[0] - prev_center[0])**2 + (cur_center[1] - prev_center[1])**2)
        prev_center = cur_center

        if (cnt is not None) and (dist_moved < STABILITY_DIST_THRESH) and (cooldown_timer == 0):
            stable_frame_count += 1
        else:
            stable_frame_count = 0

        stability_pct = min(1.0, stable_frame_count / AUTO_STABILITY_FRAMES)

        # 3) OSD 대시보드 렌더링 (영문 전용, 불량률 및 수율 반영)
        cv2.rectangle(display_frame, (0, 0), (w, 55), (30, 30, 30), -1)
        
        # 실제 종이 기준 수율 및 불량률 계산
        if stats["total_paper"] > 0:
            yield_rate = (stats["pass"] / stats["total_paper"]) * 100.0
            defect_rate = (stats["fail"] / stats["total_paper"]) * 100.0
        else:
            yield_rate = 0.0
            defect_rate = 0.0

        stat_text = f"PAPER: {stats['total_paper']}  |  PASS: {stats['pass']}  |  FAIL: {stats['fail']}  |  DEFECT: {defect_rate:.1f}%  |  YIELD: {yield_rate:.1f}%"
        cv2.putText(display_frame, stat_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        defect_text = f"DEFECTS -> TORN: {stats['defects']['TORN']} | WRINKLE: {stats['defects']['WRINKLED']} | STAIN: {stats['defects']['STAINED']} | SKIPPED(NOT PAPER): {stats['non_paper_skipped']}"
        cv2.putText(display_frame, defect_text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)

        # 외곽 에지 표시
        if cnt is not None:
            cv2.drawContours(display_frame, [cnt], -1, (0, 255, 255), 2)

        # 찢김 결함 부위 마킹
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

        # 하단 조작 가이드 & 게이지
        cv2.putText(display_frame, f"Geometry: {detail_str}", (20, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        bar_x, bar_y, bar_w, bar_h = 20, h - 25, 240, 14
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + int(bar_w * stability_pct), bar_y + bar_h), (0, 255, 0), -1)
        cv2.putText(display_frame, f"Auto-Inspect: {int(stability_pct*100)}%", (bar_x + bar_w + 12, bar_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Smart Paper Quality Inspector", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord('q'), ord('Q'), 27]:
            notify(
                f"\n🛑 [시스템 종료] 최종 수율 {yield_rate:.1f}%, 불량률 {defect_rate:.1f}%로 검사를 종료합니다.",
                "시스템을 종료합니다. 수고하셨습니다."
            )
            break

        # 4) 검사 실행 판정
        trigger_inspect = (stable_frame_count >= AUTO_STABILITY_FRAMES) or (key in [32, 13])

        if trigger_inspect:
            stable_frame_count = 0
            cooldown_timer = AUTO_COOLDOWN_FRAMES

            status_label = "ANALYZING..."
            box_color = (0, 165, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)
            cv2.imshow("Smart Paper Quality Inspector", display_frame)
            cv2.waitKey(1)

            notify("\n🔍 [검사 진행] 종이 상태를 분석 중입니다...", "종이 상태를 분석 중입니다.")

            # 1단계 기하학 검사 (종이 미감지 시 검사 제외 처리 -> 불량률에 미반영)
            if detail_str.startswith("NO_PAPER") or detail_str.startswith("TOO_SMALL"):
                stats["non_paper_skipped"] += 1
                status_label = "SKIPPED - NO PAPER DETECTED"
                box_color = (200, 200, 200)
                notify("⚠️ [검사 제외] 종이가 감지되지 않아 검사에서 제외되었습니다. (불량률 미반영)",
                       "종이가 감지되지 않아 검사에서 제외되었습니다. 노란색 종이를 비춰주세요.")
                continue

            # 2단계 VLM 정밀 검증 (손가락 가림 정상 허용 & 완화된 기준)
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
                            "You are a quality control inspector for yellow paper sheets.\n\n"
                            "Key Guidelines:\n"
                            "1. Human fingers or hands holding the edge/corner of the paper are completely normal. "
                            "DO NOT treat holding fingers as tears, cracks, or defects.\n"
                            "2. Small natural corner bends, minor curves, or slight tilt are ACCEPTABLE (PASS).\n"
                            "3. If the object is clearly NOT a paper sheet (e.g., face, desk background, keyboard, cup), output: NON_PAPER.\n"
                            "4. If the paper has an actual rip, jagged tear, large cut, hole, or missing chunk, output: TORN.\n"
                            "5. If heavily crushed or wrinkled, output: WRINKLED. If smudged/stained with ink/dirt, output: STAINED.\n"
                            "6. If it is a normal, usable yellow paper sheet (even when held by hands), output: PASS.\n\n"
                            "Constraint:\n"
                            "Output ONLY ONE word from: [PASS, TORN, WRINKLED, STAINED, NON_PAPER]."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Inspect this yellow paper. (Ignore holding fingers). Is it usable or defective? Choose [PASS, TORN, WRINKLED, STAINED, NON_PAPER]."
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

            # 3단계 결과 처리 (NON_PAPER는 불량률에서 제외)
            if "NON_PAPER" in decision:
                stats["non_paper_skipped"] += 1
                status_label = "SKIPPED - NOT A PAPER"
                box_color = (200, 200, 200)
                console_res = "⚠️ [검사 제외] 종이가 아닌 물체로 판단되어 통계에서 제외되었습니다."
                voice_res = "종이가 아닌 물체로 감지되어 검사에서 제외되었습니다."
            elif "PASS" in decision:
                stats["total_paper"] += 1
                stats["pass"] += 1
                status_label = "PASS - USABLE INTACT PAPER"
                box_color = (0, 255, 0)
                console_res = "📄 [판정 결과] 정상 (PASS - 온전한 사용 가능 종이)"
                voice_res = "검사 결과, 종이가 온전하여 정상입니다. 사용 가능합니다."
            elif "WRINKLED" in decision:
                stats["total_paper"] += 1
                stats["fail"] += 1
                stats["defects"]["WRINKLED"] += 1
                status_label = "FAIL - WRINKLED"
                box_color = (0, 0, 255)
                console_res = "📄 [판정 결과] 불량 (심한 구김 결함 - WRINKLED)"
                voice_res = "검사 결과, 종이가 심하게 구겨져 있어 불량입니다."
            elif "STAINED" in decision:
                stats["total_paper"] += 1
                stats["fail"] += 1
                stats["defects"]["STAINED"] += 1
                status_label = "FAIL - STAINED"
                box_color = (0, 0, 255)
                console_res = "📄 [판정 결과] 불량 (표면 오염/얼룩 결함 - STAINED)"
                voice_res = "검사 결과, 종이 표면에 얼룩이나 오염이 있어 불량입니다."
            else:  # TORN 또는 기타 파손
                stats["total_paper"] += 1
                stats["fail"] += 1
                stats["defects"]["TORN"] += 1
                status_label = "FAIL - TORN / CRACKED"
                box_color = (0, 0, 255)
                console_res = "📄 [판정 결과] 불량 (찢어짐/파손 결함 - TORN)"
                voice_res = "검사 결과, 종이에 찢김 결함이 발견되어 불량입니다."

            notify(console_res, voice_res)

finally:
    session_end_dt = datetime.now()
    session_end_str = session_end_dt.strftime("%Y-%m-%d %H:%M:%S")

    # ==================== 6. 누적 JSON 파일 저장 ====================
    total_paper = stats["total_paper"]
    final_pass = stats["pass"]
    final_fail = stats["fail"]
    final_yield = (final_pass / total_paper * 100.0) if total_paper > 0 else 0.0
    final_defect_rate = (final_fail / total_paper * 100.0) if total_paper > 0 else 0.0

    session_record = {
        "session_start_time": session_start_str,
        "session_end_time": session_end_str,
        "total_inspected_papers": total_paper,
        "pass_count": final_pass,
        "fail_count": final_fail,
        "defect_rate_percent": round(final_defect_rate, 2),
        "yield_rate_percent": round(final_yield, 2),
        "defect_breakdown": {
            "TORN": stats["defects"]["TORN"],
            "WRINKLED": stats["defects"]["WRINKLED"],
            "STAINED": stats["defects"]["STAINED"]
        },
        "ignored_non_paper_count": stats["non_paper_skipped"]
    }

    # 기존 JSON 데이터 로드 및 새 세션 데이터 추가
    history_data = []
    if os.path.exists(JSON_HISTORY_PATH):
        try:
            with open(JSON_HISTORY_PATH, "r", encoding="utf-8") as f:
                history_data = json.load(f)
                if not isinstance(history_data, list):
                    history_data = [history_data]
        except Exception as e:
            print(f"[JSON 로드 경고] 기존 파일을 읽을 수 없어 새로 생성합니다: {e}")
            history_data = []

    history_data.append(session_record)

    # JSON 저장
    with open(JSON_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=4)

    print("\n=======================================================")
    print(f"📊 [세션 검사 결과 JSON 저장 완료]: {JSON_HISTORY_PATH}")
    print(json.dumps(session_record, ensure_ascii=False, indent=4))
    print("=======================================================")

    cap.release()
    cv2.destroyAllWindows()

    if 'llm' in locals():
        del llm
    if 'chat_handler' in locals():
        del chat_handler
    gc.collect()
