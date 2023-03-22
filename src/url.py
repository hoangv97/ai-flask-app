import os
from typing import List, Literal

import requests
from langchain import OpenAI
from llama_index import (
    Document,
    GPTSimpleVectorIndex,
    LLMPredictor,
    NotionPageReader,
    PromptHelper,
)
from newspaper import Article
from langchain.chat_models import ChatOpenAI

from .youtube import get_documents as get_youtube_documents
from .youtube import get_youtube_video_id

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
        # llm=OpenAI(
        #     temperature=0,
        #     model_name=model_name or "gpt-3.5-turbo",
        #     max_tokens=num_outputs,
        #     openai_api_key=OPENAI_API_KEY,
        # ),
        llm=ChatOpenAI(
            temperature=0,
            openai_api_key=OPENAI_API_KEY, 
            max_tokens=num_outputs,
        ),
    )

    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    return index


def get_notion_documents(
    page_ids: List[str],
):
    reader = NotionPageReader(integration_token=NOTION_API_KEY)
    documents = reader.load_data(page_ids=page_ids)

    return documents


def handle_url(
    url: str,
    prompt: str,
    prompt_type: Literal["summarize", "qa"],
    model_name: str,
):
    try:
        video_id = get_youtube_video_id(url)
        if video_id:
            documents = get_youtube_documents(ids=[video_id], languages=["en", "vi"])

        else:
            # normal URL
            item = get_notion_item(url)

            if not item:
                article = Article(url)
                article.download()
                article.parse()
                # article.nlp()

                item = create_notion_item(article)

            documents = get_notion_documents([item["id"]])

        index = get_index(documents, model_name)

        response_mode = "tree_summarize" if prompt_type == "summarize" else "default"
        response = index.query(prompt + "\n", response_mode=response_mode)
        return response.response.strip()
    except Exception as e:
        print(e)
        return None
