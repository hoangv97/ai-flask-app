import os
import traceback
from typing import List

from langchain.agents import initialize_agent, load_tools, Tool
from langchain.chat_models import ChatOpenAI
from langchain.utilities import PythonREPL

# Link: https://python.langchain.com/en/latest/modules/agents/tools/getting_started.html

python_repl_description = 'A Python shell. Use this to execute python commands. Input should be a valid python command. If you expect output it should be printed out.'

AVAILABLE_TOOLS = [
    dict(
        name='python_repl', 
        description=python_repl_description
    ),
    dict(
        name='serpapi', 
        description='A search engine. Useful for when you need to answer questions about current events. Input should be a search query.'
    ),
    dict(
        name='wolfram-alpha', 
        description='A wolfram alpha search engine. Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life. Input should be a search query.'
    ),
    dict(
        name='requests',
        description='A portal to the internet. Use this when you need to get specific content from a site. Input should be a specific url, and the output will be all the text on that page.'
    ),
    dict(
        name='llm-math',
        description='Useful for when you need to answer questions about math.'
    ),
    dict(
        name='open-meteo-api',
        description='Useful for when you want to get weather information from the OpenMeteo API. The input should be a question in natural language that this API can answer.'
    ),
    dict(
        name='news-api',
        description='Use this when you want to get information about the top headlines of current news stories. The input should be a question in natural language that this API can answer.'
    ),
    dict(
        name='tmdb-api',
        description='Useful for when you want to get information from The Movie Database. The input should be a question in natural language that this API can answer.'
    ),
    dict(
        name='google-search',
        description='A wrapper around Google Search. Useful for when you need to answer questions about current events. Input should be a search query.'
    ),
    dict(
        name='wikipedia',
        description='A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, historical events, or other subjects. Input should be a search query.'
    ),
    dict(
        name='podcast-api',
        description='Use the Listen Notes Podcast API to search all podcasts or episodes. The input should be a question in natural language that this API can answer.'
    ),
]

AVAILABLE_TOOL_NAMES = [tool['name'] for tool in AVAILABLE_TOOLS]

def handle_prompt(prompt: str, tool_names: List[str]):
    try:
        llm = ChatOpenAI(temperature=0)
    
        tools = load_tools(tool_names, 
                        llm=llm, 
                        news_api_key=os.getenv('NEWS_API_KEY'),
                        listen_api_key=os.getenv('LISTEN_API_KEY'),
                        tmdb_bearer_token=os.getenv('TMDB_BEARER_TOKEN'),
                        )
        
        if 'python_repl' in tool_names:
            python_repl = PythonREPL()
            tools.append(
                Tool(
                    name='Python REPL',
                    func=python_repl.run,
                    description=python_repl_description,
                )
            )

        agent = initialize_agent(tools, 
                                llm, 
                                agent="zero-shot-react-description", 
                                verbose=True, 
                                return_intermediate_steps=True)
        
        result = agent({'input': prompt})
        result['success'] = True

        return result
    except Exception as e:
        traceback.print_exc()
        return {
            'success': False,
            'error': e.args,
        }
