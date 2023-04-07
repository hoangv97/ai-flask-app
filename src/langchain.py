import traceback
from typing import List

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI

from .tools import get_tools
from .utils.helper import encode_protected_output


def handle_prompt(prompt: str, tool_names: List[str]):
    try:
        llm = ChatOpenAI(temperature=0)

        tools = get_tools(tool_names, llm=llm)

        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            return_intermediate_steps=True,
        )

        result = agent({"input": encode_protected_output(prompt)})
        result["success"] = True

        return result
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": e.args,
        }
