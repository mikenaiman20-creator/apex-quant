from __future__ import annotations

import html
import ipaddress
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

log = logging.getLogger("apex.research.ingestion")

DEFAULT_FEEDS = (
    "https://rss.arxiv.org/rss/q-fin.TR",
    "https://rss.arxiv.org/rss/q-fin.ST",
)
MAX_FEED_BYTES = 2_000_000
MAX_DOCUMENT_CHARS = 15_000
MAX_ITEMS_PER_FEED = 10
MAX_ARTICLE_PAGES_PER_FEED = 5


class SourcesUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ResearchDocument:
    title: str
    source_url: str
    content: str
    source_type: str


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.casefold() in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if text:
            self.parts.append(text)


def _plain_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    return html.unescape(" ".join(parser.parts))[:MAX_DOCUMENT_CHARS]


def _validated_url(url: str) -> str:
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").casefold()
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or hostname in {"localhost", "localhost.localdomain"}
        or (address is not None and not address.is_global)
    ):
        raise ValueError(f"Unsupported research source URL: {url!r}")
    return url.strip()


def _element_text(parent: ET.Element, names: tuple[str, ...]) -> str:
    for element in parent.iter():
        local_name = element.tag.rsplit("}", 1)[-1].casefold()
        if local_name in names and element.text:
            return element.text.strip()
    return ""


def parse_feed(content: bytes, source_url: str) -> list[ResearchDocument]:
    root = ET.fromstring(content)
    documents = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1].casefold() not in {"item", "entry"}:
            continue
        title = _element_text(element, ("title",))
        link = _element_text(element, ("link", "id"))
        for child in element:
            if child.tag.rsplit("}", 1)[-1].casefold() == "link":
                link = child.attrib.get("href", link)
                if child.attrib.get("rel", "alternate") == "alternate":
                    break
        summary = _element_text(element, ("summary", "description", "encoded", "content"))
        if not title or not summary:
            continue
        try:
            item_url = _validated_url(link or source_url)
        except ValueError:
            log.warning("Skipping feed item with invalid source URL from %s.", source_url)
            continue
        documents.append(ResearchDocument(
            title=_plain_text(title)[:300],
            source_url=item_url,
            content=_plain_text(summary),
            source_type="rss",
        ))
        if len(documents) >= MAX_ITEMS_PER_FEED:
            break
    return documents


def fetch_feed(url: str, timeout: float = 15.0) -> list[ResearchDocument]:
    source_url = _validated_url(url)
    request = Request(source_url, headers={"User-Agent": "ApexQuant-Research/1.0"})
    with urlopen(request, timeout=timeout) as response:
        content = response.read(MAX_FEED_BYTES + 1)
    if len(content) > MAX_FEED_BYTES:
        raise ValueError(f"Feed exceeds {MAX_FEED_BYTES} byte limit: {source_url}")
    return parse_feed(content, source_url)


def fetch_web_article(document: ResearchDocument, timeout: float = 15.0) -> ResearchDocument | None:
    request = Request(
        _validated_url(document.source_url),
        headers={"User-Agent": "ApexQuant-Research/1.0", "Accept": "text/html"},
    )
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "").casefold()
        if content_type and "html" not in content_type:
            return None
        content = response.read(MAX_FEED_BYTES + 1)
    if len(content) > MAX_FEED_BYTES:
        log.warning("Skipping oversized research article %s.", document.source_url)
        return None
    article_text = _plain_text(content.decode("utf-8", "replace"))
    if not article_text:
        return None
    return ResearchDocument(
        document.title, document.source_url, article_text, "web_article",
    )


def _parse_vtt(content: str) -> str:
    transcript = []
    previous = ""
    for line in content.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT" or "-->" in line or line.isdigit():
            continue
        line = re.sub(r"<[^>]*>", "", line)
        line = html.unescape(re.sub(r"\s+", " ", line)).strip()
        if line and line != previous:
            transcript.append(line)
            previous = line
    return " ".join(transcript)[:MAX_DOCUMENT_CHARS]


def fetch_youtube_transcript(url: str, timeout: float = 45.0) -> ResearchDocument | None:
    video_url = _validated_url(url)
    hostname = urlparse(video_url).hostname.casefold()
    if hostname not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
        raise ValueError("Transcript URLs must use youtube.com or youtu.be.")
    executable = shutil.which("yt-dlp")
    if not executable:
        log.warning("Skipping YouTube source; optional yt-dlp executable is not installed.")
        return None
    with tempfile.TemporaryDirectory(prefix="apex-youtube-") as directory:
        output = str(Path(directory) / "%(id)s.%(ext)s")
        command = [
            executable, "--skip-download", "--write-auto-subs", "--write-subs",
            "--sub-langs", "en", "--sub-format", "vtt", "--output", output, video_url,
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"yt-dlp transcript extraction failed: {result.stderr[-500:].strip()}"
            )
        subtitle_files = sorted(Path(directory).glob("*.vtt"))
        if not subtitle_files:
            log.warning("No English transcript was available for %s.", video_url)
            return None
        transcript = _parse_vtt(subtitle_files[0].read_text(encoding="utf-8"))
        if not transcript:
            return None
        title = subtitle_files[0].stem
        return ResearchDocument(title, video_url, transcript, "youtube_transcript")


def configured_sources() -> tuple[list[str], list[str]]:
    feeds = list(DEFAULT_FEEDS)
    feeds.extend(
        value.strip()
        for value in os.getenv("APEX_RESEARCH_FEEDS", "").split(",")
        if value.strip()
    )
    videos = [
        value.strip()
        for value in os.getenv("APEX_RESEARCH_YOUTUBE_URLS", "").split(",")
        if value.strip()
    ]
    return list(dict.fromkeys(feeds)), list(dict.fromkeys(videos))


def harvest(
    feed_urls: Iterable[str] | None = None,
    video_urls: Iterable[str] | None = None,
) -> list[ResearchDocument]:
    default_feeds, default_videos = configured_sources()
    feeds = list(default_feeds if feed_urls is None else feed_urls)
    videos = list(default_videos if video_urls is None else video_urls)
    documents: dict[str, ResearchDocument] = {}
    attempted_sources = 0
    succeeded_sources = 0
    for url in feeds:
        attempted_sources += 1
        try:
            feed_documents = fetch_feed(url)
            succeeded_sources += 1
            for index, document in enumerate(feed_documents):
                if index < MAX_ARTICLE_PAGES_PER_FEED:
                    try:
                        document = fetch_web_article(document) or document
                    except (URLError, TimeoutError, OSError, ValueError) as exc:
                        log.warning("Article fetch unavailable; using feed excerpt %s: %s",
                                    document.source_url, exc)
                documents.setdefault(document.source_url, document)
        except (
            URLError, TimeoutError, OSError, ValueError, ET.ParseError,
            subprocess.SubprocessError,
        ) as exc:
            log.warning("Research feed unavailable; skipping %s: %s", url, exc)
    for url in videos:
        attempted_sources += 1
        try:
            document = fetch_youtube_transcript(url)
            succeeded_sources += 1
            if document:
                documents.setdefault(document.source_url, document)
        except (
            URLError, TimeoutError, OSError, ValueError, RuntimeError,
            subprocess.SubprocessError,
        ) as exc:
            log.warning("YouTube transcript unavailable; skipping %s: %s", url, exc)
    if attempted_sources and not succeeded_sources:
        raise SourcesUnavailable("All configured research sources were unavailable.")
    return list(documents.values())
