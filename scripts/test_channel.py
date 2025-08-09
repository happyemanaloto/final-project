import os, sys
# ensure the src/ folder is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# scripts/test_channel.py
from backend.scrapers.video_scraper import scrape_channel_recipes

if __name__ == "__main__":
    url = "https://www.youtube.com/@MarionsKitchen"
    recipes = scrape_channel_recipes(url)
    print("Found videos:", list(recipes.keys()))
