from transformers import pipeline


def summarize(text: str, min_length: int = 30, max_length: int = 130):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
