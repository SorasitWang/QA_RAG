"""
SearchData handles loading raw source files from the file system and
converting them into Source objects that the rest of the pipeline can process.
Add new fetch functions here (e.g. fetchDataFromWeb) following the same
Source-based return contract so callers stay decoupled from the data origin.
"""

import html
import os
import re

from Config import Config
from Dto.Source import Source, SourceType


_URL_PATTERN   = re.compile(r'(https?://|www\.)\S+', re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}')
_PHONE_PATTERN = re.compile(r'\+?\d[\d\s\-(). ]{6,}\d')
_HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
_BLOCK_TAG_PATTERN = re.compile(r'<\s*(br|/p|/div|/li|/h[1-6])\s*/?>', re.IGNORECASE)


def cleanText(text: str) -> str:
    """
    Remove noise lines from raw scraped text before chunking.
    Drops any line that is shorter than MinLineLength or matches a URL,
    email address, or phone number pattern.
    """
    cleaned: list[str] = []
    for line in text.splitlines():
        stripped: str = line.strip()
        if len(stripped) < Config.Model.MinLineLength:
            continue
        if _URL_PATTERN.search(stripped):
            continue
        if _EMAIL_PATTERN.search(stripped):
            continue
        if _PHONE_PATTERN.search(stripped):
            continue
        cleaned.append(stripped)
    return '\n'.join(cleaned)


def cleanFeedText(text: str) -> str:
    """
    Clean text from news, blog and SNS feeds before chunking.
    Unlike cleanText, short lines are kept and URLs are removed from a line rather
    than dropping the whole line: a social post is often one short line with a link.
    HTML markup is stripped and entities are decoded.
    """
    text = _BLOCK_TAG_PATTERN.sub('\n', text)
    text = html.unescape(_HTML_TAG_PATTERN.sub(' ', text))
    cleaned: list[str] = []
    for line in text.splitlines():
        stripped: str = re.sub(r'\s+', ' ', _URL_PATTERN.sub('', line)).strip()
        if stripped:
            cleaned.append(stripped)
    return '\n'.join(cleaned)


def sourceTitle(filePath: str) -> str:
    """
    Title used to identify a file in every store: the path without its extension.
    Kept identical to the historical format so existing indexes remain valid.
    """
    fileName, _ = os.path.splitext(filePath)
    return fileName


def loadSource(filePath: str) -> Source:
    """Read a single file and wrap its content in a Source object."""
    with open(filePath, 'r') as f:
        data = f.read()
    return Source(
        sourceTitle(filePath),
        SourceType.File.value,
        data,
        []
    )


def fetchDataFromFolder(folderPath: str) -> list[Source]:
    """
    Reads every file in folderPath and wraps its content in a Source object.
    The full path (without extension) is used as the title so duplicate
    detection in RagService can identify sources by their file path.
    Tags and optional metadata are left empty; callers can populate them
    after loading if needed.
    """
    sourceObjectList: list[Source] = []
    fileList: list[str] = os.listdir(folderPath)
    for file in fileList:
        sourceObjectList.append(loadSource(os.path.join(folderPath, file)))

    return sourceObjectList
