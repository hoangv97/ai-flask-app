from flask import Flask, request
from dotenv import load_dotenv
from src.url import handle_url

load_dotenv()

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/api/url', methods=['POST', 'GET'])
def api_url():
    url = request.args.get('url', None)
    prompt = request.args.get('p', None)
    prompt_type = request.args.get('t', '')
    model_name = request.args.get('m', None)
    if not url:
        return 'URL is missing!', 400
    if not prompt:
        return 'Prompt is missing!', 400
    
    result = handle_url(url, prompt, prompt_type, model_name)
    if not result:
        return 'Error when processing!', 500
    
    return {
        'url': url,
        'result': result,
    }
