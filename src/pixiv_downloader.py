import asyncio
import os
import re
import subprocess
import sys
import urllib.parse
from base64 import urlsafe_b64encode
from hashlib import sha256
from random import uniform
from secrets import token_urlsafe
from typing import Iterable, List, Optional
from urllib.parse import urlencode

from dotenv import load_dotenv
from gppt import GetPixivToken
import nest_asyncio
from pixivpy3 import AppPixivAPI

load_dotenv()

# Fix for Windows + Python 3.14: use ProactorEventLoop for subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Allow nested event loops (needed for Streamlit + asyncio on Windows)
nest_asyncio.apply()


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def create_pixiv_api(refresh_token: str) -> AppPixivAPI:
    aapi = AppPixivAPI()
    aapi.auth(refresh_token=refresh_token)
    return aapi


def fetch_refresh_token() -> str:
    username = os.getenv("PIXIV_USERNAME")
    password = os.getenv("PIXIV_PASSWORD")
    
    if not username or not password:
        raise ValueError(
            "Pixiv credentials required. Set PIXIV_USERNAME and PIXIV_PASSWORD "
            "in .env file or provide them as arguments."
        )
    
    g = GetPixivToken(headless=True)
    refresh_token = asyncio.run(g.login(username=username, password=password))["refresh_token"]
    return refresh_token


def sanitize_filename(name: str, max_len: int = 160) -> str:
    return _SAFE_FILENAME_RE.sub("_", name)[:max_len]


def _select_size(image_urls, size: str) -> Optional[str]:
    if size == "original" and hasattr(image_urls, "original"):
        return image_urls.original
    if size == "large" and hasattr(image_urls, "large"):
        return image_urls.large
    if size == "medium" and hasattr(image_urls, "medium"):
        return image_urls.medium
    return (
        getattr(image_urls, "large", None)
        or getattr(image_urls, "original", None)
        or getattr(image_urls, "medium", None)
    )


def get_illust_image_urls(illust, size: str = "original", first_only: bool = False) -> List[str]:
    if getattr(illust, "type", None) == "ugoira":
        return []

    urls: List[str] = []
    if getattr(illust, "meta_pages", None):
        for page in illust.meta_pages:
            url = _select_size(page.image_urls, size)
            if url:
                urls.append(url)
    else:
        meta_single = getattr(illust, "meta_single_page", None)
        original_single = getattr(meta_single, "original_image_url", None) if meta_single else None
        if original_single:
            urls.append(original_single)
        else:
            image_urls = getattr(illust, "image_urls", None)
            if image_urls is not None:
                url = _select_size(image_urls, size)
                if url:
                    urls.append(url)
    if first_only:
        return urls[:1]
    return urls


def filter_illusts(illusts: List) -> List:
    blocked_types = {"manga", "novel", "ugoira"}
    filtered = []
    for illust in illusts:
        if getattr(illust, "type", None) in blocked_types:
            continue
        if getattr(illust, "visible", True) is False:
            continue
        filtered.append(illust)
    return filtered


def _extract_offset(next_url: Optional[str]) -> Optional[int]:
    if not next_url:
        return None
    parsed = urllib.parse.urlparse(next_url)
    qs = urllib.parse.parse_qs(parsed.query)
    offset = qs.get("offset", [None])[0]
    if offset is None:
        return None
    try:
        return int(offset)
    except ValueError:
        return None


def collect_illusts(fetch_fn, base_params: dict, max_items: int, max_pages: int) -> List:
    items: List = []
    offset: Optional[int] = None
    pages = 0

    while True:
        params = dict(base_params)
        if offset is not None:
            params["offset"] = offset

        response = fetch_fn(**params)
        illusts = getattr(response, "illusts", []) or []
        items.extend(illusts)
        pages += 1

        if len(items) >= max_items:
            return items[:max_items]
        if pages >= max_pages:
            return items

        offset = _extract_offset(getattr(response, "next_url", None))
        if offset is None:
            return items


def build_search_word(word: str, include_r18: bool) -> str:
    word = (word or "").strip()
    if include_r18 and "R-18" not in word:
        word = f"{word} R-18".strip()
    if not include_r18:
        if "-R-18" not in word:
            word = f"{word} -R-18".strip()
        if "-R-18G" not in word:
            word = f"{word} -R-18G".strip()
    return word


def search_illusts(
    aapi: AppPixivAPI,
    word: str,
    search_target: str,
    sort: str,
    duration: Optional[str],
    search_ai_type: Optional[int],
    max_items: int,
    max_pages: int,
) -> List:
    params = {
        "word": word,
        "search_target": search_target,
        "sort": sort,
        "filter": "for_ios",
    }
    if duration:
        params["duration"] = duration
    if search_ai_type is not None:
        params["search_ai_type"] = search_ai_type

    return collect_illusts(aapi.search_illust, params, max_items, max_pages)


def ranking_illusts(
    aapi: AppPixivAPI,
    mode: str,
    date: Optional[str],
    max_items: int,
    max_pages: int,
) -> List:
    params = {
        "mode": mode,
        "filter": "for_ios",
    }
    if date:
        params["date"] = date

    return collect_illusts(aapi.illust_ranking, params, max_items, max_pages)


def download_illust(
    aapi: AppPixivAPI,
    illust,
    output_dir: str,
    size: str = "original",
    first_only: bool = False,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)

    urls: List[str] = []
    try:
        detail = aapi.illust_detail(illust.id)
        detail_illust = getattr(detail, "illust", None)
        if detail_illust is None:
            raise RuntimeError("missing illust detail")
        if getattr(detail_illust, "visible", True) is False:
            raise RuntimeError("not visible or restricted")
        urls = get_illust_image_urls(detail_illust, size=size, first_only=first_only)
        if not urls:
            raise RuntimeError("image urls missing")
    except Exception as exc:
        urls = get_illust_image_urls(illust, size=size, first_only=first_only)
        if not urls:
            raise RuntimeError(f"illust_detail failed: {exc}")
    if not urls:
        return []

    paths: List[str] = []
    for idx, url in enumerate(urls):
        ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
        base_name = f"{illust.id}_p{idx}{ext}"
        fname = sanitize_filename(base_name)
        aapi.download(url, path=output_dir, fname=fname)
        paths.append(os.path.join(output_dir, fname))
    return paths
