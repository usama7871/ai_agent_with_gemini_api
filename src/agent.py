# src/agent.py

from typing import List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.memory import BaseMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from src.config import AGENT_SYSTEM_PROMPT
from src.utils import logger
from langchain.schema.runnable import Runnable
# from langchain.agents.format_scratchpad import format_to_messages
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser


class AIAgent:
    """
    Orchestrates the LLM, tools, and memory to create a Langchain agent.
    """
    def __init__(self, llm: BaseChatModel, tools: List[BaseTool], memory: BaseMemory):
        """
        Initializes the AIAgent.

        Args:
            llm (BaseChatModel): The language model instance.
            tools (List[BaseTool]): A list of tools the agent can use.
            memory (BaseMemory): The memory system for conversational context.
        """
        self._llm = llm
        self._tools = tools
        self._memory = memory
        self._agent_executor: AgentExecutor | None = None
        logger.info("AIAgent initialized.")

    def _create_agent_prompt(self) -> PromptTemplate:
        """
        Creates the prompt template for the agent.
        """
        # The prompt needs to know about tools, input, and chat history.
        # This is a standard ReAct prompt structure.
        prompt = PromptTemplate.from_template(
            template=AGENT_SYSTEM_PROMPT + """

Available Tools:
{tools}

To use a tool, use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action

Observation: the result of the action

When you have a response, use the following format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

Chat History:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}
"""
        )
        logger.debug("Agent prompt created.")
        return prompt

    def get_runnable_agent(self) -> Runnable:
        """
        Creates and returns the Langchain Runnable agent.

        Returns:
            Runnable: The Langchain agent ready to be invoked.
        """
        if self._agent_executor is None:
            prompt = self._create_agent_prompt()

            # Create the ReAct agent
            agent = create_react_agent(self._llm, self._tools, prompt)

            # Create the agent executor
            self._agent_executor = AgentExecutor(
                agent=agent,
                tools=self._tools,
                memory=self._memory,
                verbose=True,     #Set to True for detailed agent trace in console
                handle_parsing_errors=True,
                max_iterations=7, # Limit tool usage to prevent infinite loops
                early_stopping_method="generate"
            )
            logger.info("Langchain AgentExecutor created.")
        return self._agent_executor
