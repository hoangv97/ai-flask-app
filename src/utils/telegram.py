import requests


def send_telegram_message(bot_id: str, chat_id: str, message: str):
    url = "https://api.telegram.org/bot{}/sendMessage".format(bot_id)
    data = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }
    requests.post(url, data=data)
