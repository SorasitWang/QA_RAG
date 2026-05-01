"""
SearchData handles loading raw source files from the file system and
converting them into Source objects that the rest of the pipeline can process.
Add new fetch functions here (e.g. fetchDataFromWeb) following the same
Source-based return contract so callers stay decoupled from the data origin.
"""

import os
import re

from Config import Config
from Dto.Source import Source


_URL_PATTERN   = re.compile(r'(https?://|www\.)\S+', re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}')
_PHONE_PATTERN = re.compile(r'\+?\d[\d\s\-(). ]{6,}\d')


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
        fullPath = os.path.join(folderPath, file)
        fileName, _ = os.path.splitext(fullPath)
        with open(fullPath, 'r') as f:
            data = f.read()
            sourceObject = Source(
                fileName,
                'text',
                data,
                []
            )
            sourceObjectList.append(sourceObject)

    return sourceObjectList
