from langchain.utilities import PythonREPL
from langchain.agents import Tool, load_tools
from .replicate import tools as replicate_tools
from typing import List
import os

# Custom tools
python_repl_util = PythonREPL()
python_repl = Tool(
    name="python-repl",
    func=python_repl_util.run,
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you expect output it should be printed out.",
)

# Link: https://python.langchain.com/en/latest/modules/agents/tools/getting_started.html

DEFAULT_TOOLS = [
    dict(
        name="serpapi",
        description="A search engine. Useful for when you need to answer questions about current events. Input should be a search query.",
    ),
    dict(
        name="wolfram-alpha",
        description="A wolfram alpha search engine. Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life. Input should be a search query.",
    ),
    dict(
        name="requests",
        description="A portal to the internet. Use this when you need to get specific content from a site. Input should be a specific url, and the output will be all the text on that page.",
    ),
    dict(
        name="llm-math",
        description="Useful for when you need to answer questions about math.",
    ),
    dict(
        name="open-meteo-api",
        description="Useful for when you want to get weather information from the OpenMeteo API. The input should be a question in natural language that this API can answer.",
    ),
    dict(
        name="news-api",
        description="Use this when you want to get information about the top headlines of current news stories. The input should be a question in natural language that this API can answer.",
    ),
    dict(
        name="tmdb-api",
        description="Useful for when you want to get information from The Movie Database. The input should be a question in natural language that this API can answer.",
    ),
    dict(
        name="google-search",
        description="A wrapper around Google Search. Useful for when you need to answer questions about current events. Input should be a search query.",
    ),
    dict(
        name="wikipedia",
        description="A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, historical events, or other subjects. Input should be a search query.",
    ),
    dict(
        name="podcast-api",
        description="Use the Listen Notes Podcast API to search all podcasts or episodes. The input should be a question in natural language that this API can answer.",
    ),
]

DEFAULT_TOOL_NAMES = [tool["name"] for tool in DEFAULT_TOOLS]

CUSTOM_TOOLS = [python_repl] + replicate_tools

AVAILABLE_TOOLS = DEFAULT_TOOLS + [
    dict(name=tool.name, description=tool.description) for tool in CUSTOM_TOOLS
]


def get_tools(tool_names: List[str], llm: any):
    tools = []

    default_tool_names = []

    for tool_name in tool_names:
        if tool_name in DEFAULT_TOOL_NAMES:
            default_tool_names.append(tool_name)
        else:
            for custom_tool in CUSTOM_TOOLS:
                if tool_name == custom_tool.name:
                    tools.append(custom_tool)

    if default_tool_names:
        tools += load_tools(
            default_tool_names,
            llm=llm,
            news_api_key=os.getenv("NEWS_API_KEY"),
            listen_api_key=os.getenv("LISTEN_API_KEY"),
            tmdb_bearer_token=os.getenv("TMDB_BEARER_TOKEN"),
        )

    return tools
