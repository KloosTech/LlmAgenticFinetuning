"""LangChain ReAct agent wired to the local AgentForge inference server."""

from __future__ import annotations

from agentforge.config import AppConfig
from agentforge.agents.tools import get_tools


_SYSTEM_PROMPT = """\
You are AgentForge, a capable AI assistant running on a locally fine-tuned model.
You have access to tools to help you complete tasks.
Think step by step. When you have gathered enough information, provide a final answer.
"""


class AgentForgeAgent:
    """LangChain ReAct agent backed by the local OpenAI-compatible server.

    Uses ``langchain_openai.ChatOpenAI`` pointed at ``http://127.0.0.1:8080/v1``
    (or whatever ``config.agent.api_base`` specifies) so any model served by
    ``agentforge serve`` works automatically.
    """

    def __init__(self, config: AppConfig) -> None:
        from langchain_openai import ChatOpenAI  # type: ignore[import]
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain import hub
        from langchain_core.prompts import ChatPromptTemplate

        self.cfg = config.agent

        self.llm = ChatOpenAI(
            base_url=self.cfg.api_base,
            api_key="local",  # required by the client but not checked by the server
            model=self.cfg.model,
            temperature=0.0,
            streaming=False,
        )

        self.tools = get_tools(self.cfg.tools)

        # Build a ReAct prompt. Try to pull from hub; fall back to a local one.
        try:
            prompt = hub.pull("hwchase17/react")
        except Exception:
            prompt = _build_react_prompt()

        react_agent = create_react_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=react_agent,
            tools=self.tools,
            verbose=self.cfg.verbose,
            max_iterations=self.cfg.max_iterations,
            handle_parsing_errors=True,
        )

    def run(self, prompt: str) -> str:
        """Run the agent on a free-text prompt and return the final answer."""
        result = self.executor.invoke({"input": prompt})
        return result.get("output", str(result))


# ---------------------------------------------------------------------------
# Fallback prompt (used when LangChain Hub is unavailable)
# ---------------------------------------------------------------------------

def _build_react_prompt():
    from langchain_core.prompts import PromptTemplate

    template = """\
{system_prompt}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    return PromptTemplate.from_template(template).partial(system_prompt=_SYSTEM_PROMPT)
