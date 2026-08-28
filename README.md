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
[ WARN:0@8.883] global cap_gstreamer.cpp:1728 open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1

========================================
🔔 [시스템 시작] 종이 상태 검사 시스템을 시작합니다.
========================================
[TTS 에러] 음성 출력 중 문제가 발생했습니다: Command '['aplay', '-D', 'plughw:2,0', 'src/audio/response.wav']' returned non-zero exit status 1.
👉 [안내] 종이를 카메라에 비추고 엔터를 누르세요. (종료는 q 입력 후 엔터)
[TTS 에러] 음성 출력 중 문제가 발생했습니다: Command '['aplay', '-D', 'plughw:2,0', 'src/audio/response.wav']' returned non-zero exit status 1.

[대기 중] Enter: 검사 시작 | q: 종료 > 

🔍 [검사 진행] 종이 상태 검사를 시작합니다. 분석 중입니다...
[TTS 에러] 음성 출력 중 문제가 발생했습니다: Command '['aplay', '-D', 'plughw:2,0', 'src/audio/response.wav']' returned non-zero exit status 1.
add_text: <|turn>system
Instruction:
주어진 이미지 속 종이의 찢김 및 훼손 여부를 확인하여 사용 가능 여부를 판단하시오.

Constraint:
종이가 온전하고 찢어지지 않았다면 반드시 '사용 가능', 찢어지거나 구멍이 났다면 반드시 '사용 불가'라고만 답하시오.
다른 단어나 문장은 절대 추가하지 마시오.<turn|>
<|turn>user
이 종이의 찢어짐 여부와 사용 가능 여부를 판정해주세요.
add_text: <|image>
add_media: preproc_out has 1 entries, grid_x = 0, grid_y = 0, has_overview = 0
image_tokens->nx = 264
image_tokens->ny = 1
batch_f32 size = 1
add_text: <image|>
add_text: <turn|>
<|turn>model

encoding image slice...
clip_encode: copying image 1/1 to input buffer (nx=1056, ny=576)
clip_encode: output embedding shape [1536, 264, 1]
image slice encoded in 1235 ms
decoding image batch 1/9, n_tokens_batch = 32
image decoded (batch 1/9) in 59 ms
decoding image batch 2/9, n_tokens_batch = 32
image decoded (batch 2/9) in 75 ms
decoding image batch 3/9, n_tokens_batch = 32
image decoded (batch 3/9) in 59 ms
decoding image batch 4/9, n_tokens_batch = 32
image decoded (batch 4/9) in 61 ms
decoding image batch 5/9, n_tokens_batch = 32
image decoded (batch 5/9) in 83 ms
decoding image batch 6/9, n_tokens_batch = 32
image decoded (batch 6/9) in 66 ms
decoding image batch 7/9, n_tokens_batch = 32
image decoded (batch 7/9) in 63 ms
decoding image batch 8/9, n_tokens_batch = 32
image decoded (batch 8/9) in 62 ms
decoding image batch 9/9, n_tokens_batch = 8
image decoded (batch 9/9) in 87 ms
📄 [판정 결과] 사용 가능 (상태: 온전함)
[TTS 에러] 음성 출력 중 문제가 발생했습니다: Command '['aplay', '-D', 'plughw:2,0', 'src/audio/response.wav']' returned non-zero exit status 1.

[대기 중] Enter: 검사 시작 | q: 종료 > q

🛑 [시스템 종료] 종이 상태 검사 시스템을 종료합니다.
[TTS 에러] 음성 출력 중 문제가 발생했습니다: Command '['aplay', '-D', 'plughw:2,0', 'src/audio/response.wav']' returned non-zero exit status 1.
GST_ARGUS: Cleaning up
CONSUMER: Done Success
GST_ARGUS: Done Success
Exception ignored in: <function Llama.__del__ at 0xffff555ac670>
Traceback (most recent call last):
  File "/home/bang/vision-llm/.venv/lib/python3.10/site-packages/llama_cpp/llama.py", line 2300, in __del__
  File "/home/bang/vision-llm/.venv/lib/python3.10/site-packages/llama_cpp/llama.py", line 2297, in close
  File "/usr/lib/python3.10/contextlib.py", line 584, in close
  File "/usr/lib/python3.10/contextlib.py", line 576, in __exit__
  File "/usr/lib/python3.10/contextlib.py", line 561, in __exit__
  File "/usr/lib/python3.10/contextlib.py", line 449, in _exit_wrapper
  File "/home/bang/vision-llm/.venv/lib/python3.10/site-packages/llama_cpp/llama_chat_format.py", line 3316, in mtmd_free
  File "/home/bang/vision-llm/.venv/lib/python3.10/site-packages/llama_cpp/_utils.py", line 53, in __enter__
ValueError: I/O operation on closed file

