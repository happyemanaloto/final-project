# # quick_test_transcribe.py
# from bot.tools import _transcript_via_api, _transcribe_any, _extract_video_id
# url = "https://www.youtube.com/watch?v=0CGHwukCwpA"
# vid = _extract_video_id(url)
# api_tx = _transcript_via_api(vid)
# print("API transcript chars:", 0 if not api_tx else len(api_tx))
# if not api_tx:
#     print("Trying Whisper fallback…")
#     w_tx = _transcribe_any(url)
#     print("Whisper transcript chars:", 0 if not w_tx else len(w_tx))


# from bot.tools import transcribe_youtube_best_effort
# url = "https://www.youtube.com/watch?v=0CGHwukCwpA"
# txt = transcribe_youtube_best_effort(url)
# print("chars:", 0 if not txt else len(txt))
# print(txt[:500])

# from bot.tools import transcribe_youtube_best_effort

# print("chars:", len(transcribe_youtube_best_effort(
#     "https://www.youtube.com/@MarionsKitchen/videos",
#     max_videos=1   # default, but explicit here
# )))



import whisper
import os

# Optional: pick from tiny, base, small, medium, large
model_size = os.getenv("CHEF_WHISPER_MODEL", "small")

print(f"Loading Whisper model: {model_size}")
model = whisper.load_model(model_size)

# Point to your local MP4
file_path = r"C:\Users\happy\Documents\ironhack\kusina-bot\Reverse1\final-project\tests\data\20250816-2317-38.1765352.mp4"

print(f"Transcribing {file_path} ...")
result = model.transcribe(file_path, fp16=False)  # fp16=False is safer on CPU
print("---- Transcript ----")
print(result["text"].strip())
