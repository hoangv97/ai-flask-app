from langchain.tools.plugin import AIPlugin

# Link: https://python.langchain.com/en/latest/modules/agents/tools/getting_started.html

DEFAULT_TOOLS = [
    # dict(
    #     name="serpapi",
    #     description="A search engine. Useful for when you need to answer questions about current events. Input should be a search query.",
    # ),
    dict(
        name="wolfram-alpha",
        description="A wolfram alpha search engine. Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life. Input should be a search query.",
    ),
    # dict(
    #     name="requests",
    #     description="A portal to the internet. Use this when you need to get specific content from a site. Input should be a specific url, and the output will be all the text on that page.",
    # ),
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

api_urls = [
    # "https://datasette.io/.well-known/ai-plugin.json",
    # "https://api.speak.com/.well-known/ai-plugin.json",
    # "https://www.wolframalpha.com/.well-known/ai-plugin.json",
    # "https://www.zapier.com/.well-known/ai-plugin.json",
    # "https://www.klarna.com/.well-known/ai-plugin.json",
    # "https://www.joinmilo.com/.well-known/ai-plugin.json",
    # "https://slack.com/.well-known/ai-plugin.json",
    # "https://schooldigger.com/.well-known/ai-plugin.json",
]

AI_PLUGINS = [AIPlugin.from_url(url) for url in api_urls]
