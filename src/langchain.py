import re
import traceback
from typing import Any, Dict, List, Union

from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    AgentType,
    LLMSingleActionAgent,
    Tool,
    initialize_agent,
)
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage

from .tools import get_tools
from .utils.helper import encode_protected_output


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    # Chat history
    chat_history: List[str]
    # Thoughts callback
    thoughts_cb: Any

    def __init_subclass__(
        cls, chat_history: List[str], thoughts_cb: Any, *args, **kwargs
    ) -> None:
        cls.chat_history = chat_history
        cls.thoughts_cb = thoughts_cb
        return super().__init_subclass__(*args, **kwargs)

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        if self.thoughts_cb:
            self.thoughts_cb(thoughts)

        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        # Create a chat_history variable from the chat history provided
        kwargs["chat_history"] = "\n".join(self.chat_history)

        formatted = self.template.format(**kwargs)

        return [HumanMessage(content=formatted)]


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

        action = match.group(1).strip()
        action_input = match.group(2).strip(" ").strip('"')

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)


def handle_chat_with_agents(
    prompt: str,
    chat_history: List[str],
    tool_names: List[str],
    thoughts_cb: Any = None,
):
    try:
        # Set up the base template
        template = """Act as an assistant and have a conversation with a human. Answer the following questions as best you can. You have access to the following tools:

{tools}

You must use the following format in every response:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! If you are sure you have the final answer or no action needed, respond "Final Answer: <answer>". If you are not sure, you can continue to use the tools.

{chat_history}
Question: {input}
{agent_scratchpad}"""

        llm = ChatOpenAI(temperature=0)

        tools = get_tools(tool_names, llm=llm)

        prompt_template = CustomPromptTemplate(
            chat_history=chat_history,
            thoughts_cb=thoughts_cb,
            template=template,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"],
        )

        output_parser = CustomOutputParser()

        # LLM chain consisting of the LLM and a prompt
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
        )

        tool_names = [tool.name for tool in tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            max_iterations=15,
            return_intermediate_steps=True,
            verbose=True,
        )

        result = agent_executor({"input": encode_protected_output(prompt)})
        result["success"] = True

        return result
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": e.args,
        }
