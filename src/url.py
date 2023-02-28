import os
from newspaper import Article
import requests
from llama_index import (
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
    NotionPageReader,
    Document,
    download_loader,
)
from langchain import OpenAI
from typing import List, Literal

NOTION_API_KEY = os.getenv("NOTION_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

NOTION_HEADERS = {
    "accept": "application/json",
    "Notion-Version": "2022-06-28",
    "content-type": "application/json",
    "Authorization": "Bearer {}".format(NOTION_API_KEY),
}

notion_database_id = "db0f27fb136943e3b141b1208b65580b"


def get_notion_item(url: str):
    notion_url = "https://api.notion.com/v1/databases/{}/query".format(
        notion_database_id
    )

    payload = {
        "filter": {"and": [{"property": "URL", "url": {"equals": url}}]},
        "page_size": 100,
    }

    response = requests.post(notion_url, json=payload, headers=NOTION_HEADERS)

    result = response.json()

    return None if not result["results"] else result["results"][0]


def create_notion_item(article: Article):
    url = "https://api.notion.com/v1/pages"

    def text_to_blocks():
        lines = article.text.split("\n")
        blocks = []

        for line in lines:
            if line:
                blocks.append(
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {
                                        "content": line,
                                    },
                                }
                            ]
                        },
                    }
                )

        return blocks

    payload = {
        "parent": {"database_id": notion_database_id},
        "properties": {
            "Title": {"title": [{"text": {"content": article.title}}]},
            "URL": {"url": article.url},
        },
        "children": text_to_blocks(),
    }

    response = requests.post(url, json=payload, headers=NOTION_HEADERS)
    result = response.json()
    return result


def get_index(
    documents: List[Document],
    model_name: str,
):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2056
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )

    # define LLM
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0,
            model_name=model_name or "text-davinci-003",
            max_tokens=num_outputs,
            openai_api_key=OPENAI_API_KEY,
        )
    )

    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    return index


def ask_gpt_with_notion(
    page_ids: List[str],
    prompt: str,
    response_mode: Literal["default", "compact", "tree_summarize"],
    model_name: str,
):
    reader = NotionPageReader(integration_token=NOTION_API_KEY)
    documents = reader.load_data(page_ids=page_ids)

    index = get_index(documents, model_name)

    response = index.query(prompt + "\n", response_mode=response_mode)
    return response.response.strip()


def ask_gpt_with_youtube(
    ytlinks: List[str],
    prompt: str,
    response_mode: Literal["default", "compact", "tree_summarize"],
    model_name: str,
):
    YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")

    loader = YoutubeTranscriptReader()
    documents = loader.load_data(ytlinks=ytlinks, languages=['en', 'vi'])

    index = get_index(documents, model_name)

    response = index.query(prompt + "\n", response_mode=response_mode)
    return response.response.strip()


def handle_url(
    url: str,
    prompt: str,
    prompt_type: Literal["summarize", "qa"],
    model_name: str,
):
    try:
        response_mode = "tree_summarize" if prompt_type == "summarize" else "default"

        if url.startswith("https://www.youtube.com/watch"):
            result = ask_gpt_with_youtube([url], prompt, response_mode, model_name)

        else:
            item = get_notion_item(url)

            if not item:
                article = Article(url)
                article.download()
                article.parse()
                # article.nlp()

                item = create_notion_item(article)

            result = ask_gpt_with_notion(
                [item["id"]], prompt, response_mode, model_name
            )

        return result
    except Exception as e:
        print(e)
        return None
