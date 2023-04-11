import os

from dotenv import load_dotenv
from flask import Flask, abort, request
from flask_cors import CORS

from src.cohere import summarize as summarize_cohere
from src.hugging_face import summarize as summarize_hugging_face
from src.langchain import handle_chat_with_agents
from src.llama_index import handle_url
from src.tools import get_available_tools
from src.utils.telegram import send_telegram_message

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


def check_auth():
    api_key = request.args.get("apiKey", None)
    if api_key != os.getenv("AUTH_KEY"):
        abort(401)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/api/url", methods=["POST", "GET"])
def api_url():
    check_auth()

    url = request.args.get("url", None)
    prompt = request.args.get("p", None)
    prompt_type = request.args.get("t", "")
    model_name = request.args.get("m", None)
    if not url:
        return "URL is missing!", 400
    if not prompt:
        return "Prompt is missing!", 400

    result = handle_url(url, prompt, prompt_type, model_name)
    if not result:
        return "Error when processing!", 500

    return {
        "url": url,
        "result": result,
    }


@app.route("/api/tools", methods=["POST", "GET"])
def api_tools():
    check_auth()

    return get_available_tools()


@app.route("/api/chat", methods=["POST"])
def api_chat():
    check_auth()
    prompt = request.args.get("p", None)
    if not prompt:
        prompt = request.json.get("p", None)
    if not prompt:
        return "Prompt is missing!", 400

    tool_names = request.args.get("t", "")
    if not tool_names:
        return (
            "Tools is missing! Available tools: /api/tools",
            400,
        )
    tool_names = tool_names.split(",")

    actor = request.json.get("actor", "assistant")
    chat_history = request.json.get("h", [])

    def thoughts_cb(thoughts):
        telegram = request.json.get("telegram", None)
        if telegram:
            send_telegram_message(
                bot_id=telegram["bot_id"],
                chat_id=telegram["chat_id"],
                message="""```\n{}```""".format(thoughts),
            )

    result = handle_chat_with_agents(
        prompt,
        chat_history,
        tool_names,
        actor=actor,
        thoughts_cb=thoughts_cb,
    )
    return result


@app.route("/api/summarize", methods=["POST", "GET"])
def api_summarize():
    check_auth()

    model = request.args.get("model", None)

    text = request.json.get("text", None)
    if not text:
        return "Text is missing!", 400

    result = None

    if model in ["hugging_face", "hf"]:
        min_length = request.json.get("min_length", 30)
        max_length = request.json.get("max_length", 130)

        result = summarize_hugging_face(
            text,
            min_length=min_length,
            max_length=max_length,
        )
    else:
        temperature = request.json.get("temperature", 0.5)
        length = request.json.get("length", "medium")
        format = request.json.get("format", "paragraph")

        result = summarize_cohere(
            text,
            temperature=float(temperature),
            length=length,
            format=format,
        )
    if not result:
        return "Error when processing!", 500

    return {
        "result": result,
    }
