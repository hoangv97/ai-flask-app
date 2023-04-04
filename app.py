from dotenv import load_dotenv
from flask import Flask, request

from src.url import handle_url
from src.agents import handle_prompt, AVAILABLE_TOOL_NAMES, AVAILABLE_TOOLS

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
    
@app.route('/api/tools', methods=['POST', 'GET'])
def api_tools():
    return AVAILABLE_TOOLS
    
    
@app.route('/api/ask', methods=['POST', 'GET'])
def api_ask():
    prompt = request.args.get('p', None)
    if not prompt:
        prompt = request.json.get('p', None)
    if not prompt:
        return 'Prompt is missing!', 400
    
    tools = request.args.get('t', '')
    if not tools:
        return 'Tools is missing! Available tools: {}'.format(', '.join(AVAILABLE_TOOL_NAMES)), 400
    tools = tools.split(',')
    
    result = handle_prompt(prompt, tools)
    return result
