from dotenv import load_dotenv
from langchain import (
    HuggingFaceHub,
    LLMChain,
    LLMMathChain,
    PromptTemplate,
    SerpAPIWrapper,
)
from langchain.agents import Tool, initialize_agent, tool
from langchain.llms import HuggingFacePipeline, OpenAI
from langchain.tools import BaseTool
from langchain.utilities import GoogleSearchAPIWrapper
from transformers import pipeline

load_dotenv()

template = """Classify the Sentence: {sentence}
Answer: The classification is."""

prompt_sbs = PromptTemplate(template=template, input_variables=["sentence"])

text_gen = pipeline("text-generation")

llm_classifier = HuggingFacePipeline(pipeline=text_gen)

support_llm = OpenAI(temperature=0)

search_func = GoogleSearchAPIWrapper()
classifier_chain = LLMChain(llm=llm_classifier, prompt=prompt_sbs, verbose=True)

tools = [
    Tool(
        name="Google Search",
        func=search_func.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="Classifier",
        func=classifier_chain.run,
        description="useful for when you need to Classify the sentences",
    ),
]

agent_classifier = initialize_agent(
    tools, support_llm, agent="zero-shot-react-description", verbose=True
)

agent_classifier.run(
    """Get the recent news on World Economic forum. And classify it as positive or negative"""
)

classifier = pipeline("sentiment-analysis")


@tool("classifier", return_direct=True)
def classify_query(sentence: str) -> str:
    """Classifies the sentence."""
    reply = classifier(sentence)
    return reply[0]["label"]


classify_query.run("I am super fond of that fruit.")

tools_01 = [
    Tool(
        name="Google Search",
        func=search_func.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="Classifier",
        func=classify_query.run,
        description="useful for when you need to Classify the sentences",
    ),
]

sentiment_classifier = initialize_agent(
    tools_01, support_llm, agent="zero-shot-react-description", verbose=True
)

sentiment_classifier.run(
    """Get the recent news on World Economic forum. And classify it as positive or negative"""
)
