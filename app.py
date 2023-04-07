import os

from dotenv import load_dotenv
from flask import Flask, abort, request
from flask_cors import CORS

from src.langchain import handle_prompt
from src.tools import AVAILABLE_TOOLS
from src.url import handle_url

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

    return AVAILABLE_TOOLS


@app.route("/api/ask", methods=["POST", "GET"])
def api_ask():
    check_auth()
    prompt = request.args.get("p", None)
    if not prompt:
        prompt = request.json.get("p", None)
    if not prompt:
        return "Prompt is missing!", 400

    tools = request.args.get("t", "")
    if not tools:
        return (
            "Tools is missing! Available tools: /api/tools",
            400,
        )
    tools = tools.split(",")

    result = handle_prompt(prompt, tools)
    return result
