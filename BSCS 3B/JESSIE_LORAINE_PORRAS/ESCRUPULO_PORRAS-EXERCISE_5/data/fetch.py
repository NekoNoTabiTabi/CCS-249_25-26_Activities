import re
import requests
from bs4 import BeautifulSoup

def fetch_wikipedia_article(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (SGNS-Training)"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    content_div = soup.find("div", {"id": "mw-content-text"})

    paragraphs = content_div.find_all(["p", "li"])
    text_blocks = [p.get_text(" ", strip=True) for p in paragraphs if p.get_text()]

    text = "\n".join(text_blocks)
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text