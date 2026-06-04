import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import requests

SITEMAP_URL = "https://developer.celonis.com/sitemap.xml"
CORPUS_DIR = Path("./corpus")

WHITELIST_PREFIXES = [
    "/celonis-apis/",
    "/data-ingestion-api/",
    "/process-intelligence-apis/",
    "/cpm/developer/",
]

BLACKLIST_KEYWORDS = [
    "/openapi/",
    "/changelog",
    "/contact",
    "/use-cases",
]


def parse_sitemap(url: str) -> list[str]:
    """Fetch and parse sitemap, extract all URLs."""
    response = requests.get(url)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [
        elem.text
        for elem in root.findall("ns:url/ns:loc", namespace)
        if elem.text is not None
    ]
    return urls


def should_include_url(url: str) -> bool:
    """Apply hybrid filter: blacklist bad stuff, keep good sections."""
    # Check blacklist first
    if any(keyword in url for keyword in BLACKLIST_KEYWORDS):
        return False

    # Check whitelist
    path = urlparse(url).path
    return any(prefix in path for prefix in WHITELIST_PREFIXES)


def download_url(url: str, output_path: Path) -> bool:
    """Download a URL and save as HTML. Return True if successful."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def download_corpus() -> int:
    """Download all filtered URLs to ./corpus/. Return count of downloaded files."""
    CORPUS_DIR.mkdir(exist_ok=True)

    print("Fetching sitemap...")
    urls = parse_sitemap(SITEMAP_URL)
    print(f"Total URLs in sitemap: {len(urls)}")

    filtered_urls = [url for url in urls if should_include_url(url)]
    print(f"URLs after filtering: {len(filtered_urls)}")

    downloaded = 0
    for i, url in enumerate(filtered_urls):
        # Generate filename from URL
        path = urlparse(url).path.strip("/")
        filename = path.replace("/", "_") + ".html"
        output_path = CORPUS_DIR / filename

        print(f"[{i + 1}/{len(filtered_urls)}] Downloading: {url}")
        if download_url(url, output_path):
            downloaded += 1

        # Be respectful: wait a bit between requests
        time.sleep(0.5)

    print(f"\nDownloaded {downloaded}/{len(filtered_urls)} pages to {CORPUS_DIR}/")
    return downloaded


if __name__ == "__main__":
    download_corpus()
