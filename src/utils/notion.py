import os

import requests


def query_database(database_id: str):
    url = f"https://api.notion.com/v1/databases/{database_id}/query"

    payload = {"page_size": 100}
    headers = {
        "accept": "application/json",
        "Notion-Version": "2022-06-28",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('NOTION_API_KEY')}",
    }

    response = requests.post(url, json=payload, headers=headers)

    return response.json()
