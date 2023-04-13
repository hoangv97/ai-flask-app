import os
import re
from typing import Any, List, Union

from langchain import LLMChain, OpenAI, SerpAPIWrapper
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
    Tool,
    load_tools,
)
from langchain.agents.agent_toolkits import NLAToolkit
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish, Document
from langchain.tools.plugin import AIPlugin
from langchain.utilities import PythonREPL
from langchain.vectorstores import FAISS

from ..utils.notion import query_database
from .const import DEFAULT_TOOL_NAMES, DEFAULT_TOOLS, AI_PLUGINS
from .giphy import giphy
from .replicate import tools as replicate_tools

CUSTOM_TOOLS = [
    giphy,
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


def get_tools_by_query(query: str, llm: any):
    embeddings = OpenAIEmbeddings()
    docs = []

    tools_dict = {}

    for plugin in AI_PLUGINS:
        docs.append(
            Document(
                page_content=plugin.description_for_model,
                metadata={"plugin_name": plugin.name_for_model},
            )
        )
        tools_dict[plugin.name_for_model] = NLAToolkit.from_llm_and_ai_plugin(
            llm, plugin
        ).nla_tools

    for tool in CUSTOM_TOOLS + load_default_tools(DEFAULT_TOOL_NAMES, llm):
        docs.append(
            Document(
                page_content=tool.description,
                metadata={"plugin_name": tool.name},
            )
        )
        tools_dict[tool.name] = [tool]

    vector_store = FAISS.from_documents(docs, embeddings)

    retriever = vector_store.as_retriever()

    # Get documents, which contain the Plugins to use
    result_docs = retriever.get_relevant_documents(query)

    tools = []
    for doc in result_docs:
        tool_name = doc.metadata["plugin_name"]
        if tool_name in tools_dict:
            tools.extend(tools_dict[tool_name])

    return tools
