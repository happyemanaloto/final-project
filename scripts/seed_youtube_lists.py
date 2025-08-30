from pathlib import Path; import sys, os
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from dotenv import load_dotenv; load_dotenv()

from bot.tools import summarize_video

URLS = [
 "https://www.youtube.com/watch?v=Gyz7s3cFjZU",
 "https://www.youtube.com/watch?v=NTpzPZajtEU",
 "https://www.youtube.com/watch?v=oPXfLnb8pFo",
 "https://www.youtube.com/watch?v=zZNhVv7fmSE",
 "https://www.youtube.com/watch?v=VRctr-tviIA",
 "https://www.youtube.com/watch?v=Swkq2jc5AnA",
 "https://www.youtube.com/watch?v=SkbOKonW6nU",
 "https://www.youtube.com/watch?v=K9qJQmOeohU",
 "https://www.youtube.com/watch?v=QlDzm8UXbk0",
 "https://www.youtube.com/watch?v=u8bdtAUpvlA",
]

ok = 0
for u in URLS:
    try:
        out = summarize_video.invoke({"url": u, "target_lang": "en"})
        print("OK:", u, str(out)[:120], "…")
        ok += 1
    except Exception as e:
        print("FAIL:", u, e)
print(f"Done. Seeded {ok}/{len(URLS)}.")
