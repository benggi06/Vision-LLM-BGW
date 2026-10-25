# Vision-LLM-BGW

[금오공대] **Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템** 과정(2026.08)에서 실습한 내용을 정리한 저장소입니다.
*Coursework on vision + LLM multimodal systems for Physical AI.*

## 실습 노트북

| 노트북 | 내용 |
|---|---|
| `00_Initial-Setup` | 개발 환경 설정 |
| `01_Linux-and-Python` | Linux·Python 기초 |
| `02_Computer-Vision` | OpenCV 영상 처리 |
| `03_DL-and-GPU` | 딥러닝과 GPU 연산 (CPU/GPU 비교, conv2d GPU 구현) |
| `04_DL-Object-Detection` | 객체 탐지 (YOLO, NMS) |
| `05-LLM_and_Gemma` | LLM과 Gemma |
| `06-Vision_LLM_Multimodal_Systems` | 비전 + LLM 멀티모달 시스템 |

## 스크립트

- `yolo1.py`, `object_detection.py`: YOLO 객체 탐지
- `object_tracking.py`, `object_tracking_cv.py`: 객체 추적
- `gemma.py`, `yolo_gemma.py`, `yoloy_gemma2.py`: 탐지 결과를 Gemma LLM으로 해석
- `sts.py`, `tts.py`: 음성 입력·출력
- `project.py`: 비전 + LLM + 음성을 묶은 최종 프로젝트

## 기술

Python · OpenCV · Ultralytics YOLO · Gemma · Jupyter
