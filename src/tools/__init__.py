import os
from typing import List, Any

from langchain.agents import Tool, load_tools
from langchain.utilities import PythonREPL

from .replicate import tools as replicate_tools
from .giphy import GiphyAPIWrapper
from ..utils.notion import query_database

# Custom tools
python_repl_util = PythonREPL()
python_repl = Tool(
    name="python-repl",
    func=python_repl_util.run,
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you expect output it should be printed out.",
)

giphy = Tool(
    name="giphy",
    func=GiphyAPIWrapper().run,
    description="useful for when you need to find gif or clips. The input of this tool in json format only with q key as description of the image; limit key as number of result in integer; api_type key as one of these values: search, trending; random key as user requests random results or not in boolean format.",
    return_direct=True,
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
        description="Useful for when you want to get current weather information. The input should be a question in natural language that this API can answer.",
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

CUSTOM_TOOLS = [
    giphy,
    python_repl,
] + replicate_tools


def get_notion_tools() -> dict:
    tools_dict = {}

    database_id = os.getenv("PLUGINS_NOTION_DATABASE_ID", None)

    if not database_id:
        return tools_dict

    result = query_database(database_id=database_id)
    for row in result["results"]:
        name = row["properties"]["Name"]["title"][0]["plain_text"]
        description = row["properties"]["Description"]["rich_text"][0]["plain_text"]
        groups = list(
            map(
                lambda group: group["name"], row["properties"]["Groups"]["multi_select"]
            )
        )

        tools_dict[name] = {
            "description": description,
            "groups": groups,
        }

    return tools_dict


def get_available_tools() -> List[dict]:
    tools_dict = get_notion_tools()
    tools = [
        dict(
            name=tool.name,
            description=tool.description
            if tool.name not in tools_dict
            else tools_dict[tool.name]["description"],
            groups=[]
            if tool.name not in tools_dict
            else tools_dict[tool.name]["groups"],
        )
        for tool in CUSTOM_TOOLS
    ] + [
        dict(
            name=tool["name"],
            description=tool["description"]
            if tool["name"] not in tools_dict
            else tools_dict[tool["name"]]["description"],
            groups=[]
            if tool["name"] not in tools_dict
            else tools_dict[tool["name"]]["groups"],
        )
        for tool in DEFAULT_TOOLS
    ]
    # sort tools by name
    tools = sorted(tools, key=lambda tool: tool["name"])
    return tools


def load_default_tools(tool_names: List[str], llm: any):
    default_tools = load_tools(
        tool_names,
        llm=llm,
        news_api_key=os.getenv("NEWS_API_KEY"),
        listen_api_key=os.getenv("LISTEN_API_KEY"),
        tmdb_bearer_token=os.getenv("TMDB_BEARER_TOKEN"),
    )
    tools = []
    for tool in default_tools:
        tools.append(
            Tool(
                name=tool.name,
                func=tool.run,
                description=tool.description,
            )
        )
    return tools


def get_tools(tool_names: List[str], llm: any):
    tools = []

    default_tool_names = []

    tools_dict = get_notion_tools()

    for tool_name in tool_names:
        if tool_name in DEFAULT_TOOL_NAMES:
            default_tool_names.append(tool_name)
        else:
            for custom_tool in CUSTOM_TOOLS:
                if tool_name == custom_tool.name:
                    if tool_name in tools_dict:
                        custom_tool.description = tools_dict[tool_name]["description"]
                    tools.append(custom_tool)

    if default_tool_names:
        default_tools = load_default_tools(default_tool_names, llm)
        for tool in default_tools:
            if tool.name in tools_dict:
                tool.description = tools_dict[tool.name]["description"]
            tools.append(tool)

    return tools
