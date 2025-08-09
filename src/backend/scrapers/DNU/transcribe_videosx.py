# src/backend/scrapers/simple_channel_scraper.py
"""
Robust scraper for YouTube channel videos using yt_dlp.
Fetches all video URLs from a channel or handle URL.
Usage:
    python simple_channel_scraper.py
"""
from yt_dlp import YoutubeDL

# Replace with any channel URL (handle, user, channel)
CHANNEL_URL = "https://www.youtube.com/@Marionskitchen/videos"


def fetch_video_urls(channel_url: str) -> list[str]:
    """
    Uses yt_dlp to extract video entries from a channel page.
    Returns a list of video URLs.
    """
    ydl_opts = {
        'quiet': True,
        'extract_flat': 'in_playlist',  # do not download, just metadata
        'dump_single_json': True
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    entries = info.get('entries', [])
    video_urls = []
    for entry in entries:
        url = entry.get('url') or entry.get('webpage_url')
        if url and url.startswith('http'):
            video_urls.append(url)
    return video_urls

def main():
    print(f"Scraping video URLs from: {CHANNEL_URL}\n")
    urls = fetch_video_urls(CHANNEL_URL)
    for i, url in enumerate(urls, start=1):
        print(f"{i}. {url}")


if __name__ == "__main__":
    main()