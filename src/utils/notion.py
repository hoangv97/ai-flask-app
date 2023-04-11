import os
from typing import List

import requests

url_prefix = "https://api.notion.com/v1"

headers = {
    "accept": "application/json",
    "Notion-Version": "2022-06-28",
    "content-type": "application/json",
    "Authorization": f"Bearer {os.getenv('NOTION_API_KEY')}",
}


def query_database(database_id: str, filter_data: dict = None):
    url = f"{url_prefix}/databases/{database_id}/query"

    payload = {
        "page_size": 100,
    }
    if filter_data:
        payload["filter"] = filter_data

    response = requests.post(url, json=payload, headers=headers)

    result = response.json()
    return result


def create_page(parent: dict, properties: dict, children: List[dict]):
    url = f"{url_prefix}/pages"
    payload = {
        "parent": parent,
        "properties": properties,
        "children": children,
    }

    response = requests.post(url, json=payload, headers=headers)
    result = response.json()
    return result
