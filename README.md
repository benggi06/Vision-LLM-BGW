(.venv) bang@bang-desktop:~/vision-llm$ python3 project.py 

[시스템 준비] VLM 모델을 로드하는 중입니다. 잠시만 기다려주세요...
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
llama_kv_cache: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to 512
llama_kv_cache: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to 512
GST_ARGUS: Creating output stream
CONSUMER: Waiting until producer is connected...
GST_ARGUS: Available Sensor modes :
GST_ARGUS: 3280 x 2464 FR = 21.000000 fps Duration = 47619048 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 3280 x 1848 FR = 28.000001 fps Duration = 35714284 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 1920 x 1080 FR = 29.999999 fps Duration = 33333334 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 1640 x 1232 FR = 29.999999 fps Duration = 33333334 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 1280 x 720 FR = 59.999999 fps Duration = 16666667 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: Running with following settings:
   Camera index = 0 
   Camera mode  = 4 
   Output Stream W = 1280 H = 720 
   seconds to Run    = 0 
   Frame Rate = 59.999999 
GST_ARGUS: Setup Complete, Starting captures for 0 seconds
GST_ARGUS: Starting repeat capture requests.
CONSUMER: Producer has connected; continuing.
[ WARN:0@6.542] global cap_gstreamer.cpp:1728 open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1
Gtk-Message: 16:17:28.335: Failed to load module "canberra-gtk-module"

=======================================================
🔔 [시스템 시작] 지능형 종이 품질 자동 검사 시스템을 시작합니다.
=======================================================
👉 [안내] 노란색 종이를 비추고 잠시 멈추면 자동 검사합니다. (수동: SPACE / 종료: Q)

=======================================================
📊 [세션 검사 결과 JSON 저장 완료]: src/output/inspection_history.json
{
    "session_start_time": "2026-08-28 16:17:28",
    "session_end_time": "2026-08-28 16:17:39",
    "total_inspected_papers": 0,
    "pass_count": 0,
    "fail_count": 0,
    "defect_rate_percent": 0.0,
    "yield_rate_percent": 0.0,
    "defect_breakdown": {
        "TORN": 0,
        "WRINKLED": 0,
        "STAINED": 0
    },
    "ignored_non_paper_count": 0
}
=======================================================
GST_ARGUS: Cleaning up
CONSUMER: Done Success
GST_ARGUS: Done Success
Traceback (most recent call last):
  File "/home/bang/vision-llm/project.py", line 215, in <module>
    bbox, is_geom_valid, rect_score, detail_str, cnt, defects = analyze_yellow_paper_defects_relaxed(frame)
NameError: name 'analyze_yellow_paper_defects_relaxed' is not defined. Did you mean: 'analyze_yellow_paper_relaxed'?

