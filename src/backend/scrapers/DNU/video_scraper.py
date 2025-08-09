# src/backend/scrapers/video_scraper.py
"""
Module to scrape recipe content from YouTube cooking videos and entire channels.
"""
import os
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
import json
from bs4 import BeautifulSoup


def extract_video_id(url: str) -> str:
    """
    Extracts the video ID from a YouTube URL.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if hostname.lower() == "youtu.be":
        return parsed.path.lstrip('/')
    if hostname.lower() in ("www.youtube.com", "youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith("/embed/") or parsed.path.startswith("/v/"):
            return parsed.path.split("/")[-1]
    raise ValueError(f"Invalid YouTube URL: {url}")


def scrape_video_recipe(url: str) -> dict:
    """
    Uses a video's transcript to extract a structured recipe via the LLM.
    Returns dict with keys: title, ingredients, steps, servings, cook_time
    """
    vid_id = extract_video_id(url)
    transcript_segments = YouTubeTranscriptApi.get_transcript(vid_id)
    text = "".join(seg["text"] for seg in transcript_segments)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    llm = OpenAI()
    prompt = (
        "Extract the recipe details (title, ingredients list, numbered steps, servings, cook time) "
        "from the following transcript text. Return as JSON."
        " + "
        "."join(chunks)
    )
    response = llm(prompt)
    return json.loads(response)


def scrape_channel_recipes(channel_videos_url: str) -> dict:
    """
    Scrapes recipes from every video listed on a YouTube channel's /videos page.

    Args:
        channel_videos_url: URL to the channel's /videos page (e.g. https://www.youtube.com/@Marionskitchen/videos)

    Returns:
        dict mapping each video URL to its extracted recipe dict.
    """
    # Ensure URL ends with /videos
    url = channel_videos_url.rstrip('/')
    if not url.endswith('/videos'):
        url = url + '/videos'

    # Fetch page HTML
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    # Extract unique video URLs
    hrefs = [a['href'] for a in soup.find_all('a', href=True) if '/watch?v=' in a['href']]
    video_urls = []
    for h in hrefs:
        full = h if h.startswith('http') else f"https://www.youtube.com{h}"
        if full not in video_urls:
            video_urls.append(full)

    recipes = {}
    for vurl in video_urls:
        try:
            recipes[vurl] = scrape_video_recipe(vurl)
        except Exception as e:
            print(f"Failed to scrape {vurl}: {e}")
    return recipes
