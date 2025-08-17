#!/usr/bin/env python3
import sys, os, re, whisper
from yt_dlp import YoutubeDL

def download_audio(video_url: str, output_wav: str = "audio.wav") -> str:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "audio.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "ffmpeg_location": r"C:\Users\happy\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin",
        "quiet": True,
        "no_warnings": True,
    }
    print(f"[1/4] Downloading & converting to WAV via yt-dlp…")
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    if not os.path.isfile(output_wav):
        raise RuntimeError(f"Expected {output_wav} but didn’t find it.")
    return output_wav

def transcribe_audio(wav_file: str, model_name: str = "base") -> str:
    print(f"[2/4] Loading Whisper model '{model_name}'…")
    model = whisper.load_model(model_name)
    print(f"[3/4] Transcribing '{wav_file}'…")
    return model.transcribe(wav_file)["text"]

def clean_text(text: str) -> str:
    cleaned = re.sub(r"\[.*?\]", "", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()

def main():
    if len(sys.argv) != 2:
        print("Usage: python simple_channel_scraper.py <YOUTUBE_URL>")
        sys.exit(1)

    url = sys.argv[1]
    wav = download_audio(url)
    raw = transcribe_audio(wav)
    cleaned = clean_text(raw)

    # Output & save
    print("\n===== Transcript =====\n")
    print(cleaned)
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)
    print(f"\nTranscript saved to: {os.path.abspath('transcript.txt')}")

    # Cleanup
    os.remove(wav)
    print("Temporary files removed.")

if __name__ == "__main__":
    main()
