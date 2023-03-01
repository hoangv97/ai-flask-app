import re
from typing import List

from llama_index.readers.schema.base import Document
from youtube_transcript_api import YouTubeTranscriptApi


def get_youtube_video_id(url: str):
    # Regular expression to match YouTube video URLs
    regex = r"(?:https?:\/\/)?(?:[0-9A-Z-]+\.)?(?:youtube|youtu|youtube-nocookie)\.(?:com|be)\/(?:watch\?v=|watch\?.+&v=|embed\/|v\/|.+\?v=)?([^&=\n%\?]{11})"

    match = re.match(regex, url, re.IGNORECASE)
    if match:
        return match.group(1) or match.group(2) or match.group(3)

    # If the URL is not a YouTube video, return None
    return None


def get_documents(ids: List[str], languages: List[str]):
    results = []
    for id in ids:
        srt = YouTubeTranscriptApi.get_transcript(id, languages=languages)
        transcript = ""
        for chunk in srt:
            transcript = transcript + chunk["text"] + "\n"
        results.append(Document(transcript))
    return results
