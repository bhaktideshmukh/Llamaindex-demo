from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
    ToolCall,
)
from llama_index.core.workflow import Context

from config import llm
from tools import make_notes_tool, text_tools, today_tool, web_search, web_search_tool


def build_single_agent() -> AgentWorkflow:
    """Function-calling agent (used automatically when the LLM supports it)."""
    return AgentWorkflow.from_tools_or_functions(
        [make_notes_tool(), web_search_tool, today_tool, *text_tools],
        llm=llm,
        system_prompt=(
            "You are a study assistant. Check the user's notes first; "
            "use web search only if the notes don't cover the question."
        ),
    )


async def web_search_tracked(ctx: Context, query: str) -> str:
    """Search the web for information on a topic."""
    state = await ctx.store.get("state")  # shared workflow state
    state["searches"].append(query)
    await ctx.store.set("state", state)
    return web_search(query)


def build_multi_agent() -> AgentWorkflow:
    """Two specialised agents; the root agent can hand off to the other."""
    notes_expert = FunctionAgent(
        name="notes_expert",
        description="Answers questions using the user's study notes",
        system_prompt="Answer from the notes. If they don't cover it, hand off to researcher.",
        tools=[make_notes_tool()],
        llm=llm,
        can_handoff_to=["researcher"],
    )
    researcher = FunctionAgent(
        name="researcher",
        description="Searches the web for information missing from the notes",
        system_prompt="Search the web and summarize what you find in two sentences.",
        tools=[web_search_tracked],
        llm=llm,
        can_handoff_to=["notes_expert"],
    )
    return AgentWorkflow(
        agents=[notes_expert, researcher],
        root_agent="notes_expert",
        initial_state={"searches": []},
    )


async def run_and_show_tools(handler):
    async for event in handler.stream_events():
        if isinstance(event, ToolCall):
            print(f"  [tool] {event.tool_name} {event.tool_kwargs}")
    return await handler


async def demo():
    print("== 1. Function-calling agent + memory (Context) ==")
    agent = build_single_agent()
    ctx = Context(agent)
    print(await run_and_show_tools(agent.run(user_msg="What are the five stages of RAG?", ctx=ctx)))
    print(await run_and_show_tools(
        agent.run(user_msg="Summarize that in one sentence and tell me its word count.", ctx=ctx)
    ))

    print("\n== 2. ReAct agent (reasons in text) ==")
    react = ReActAgent(tools=[make_notes_tool()], llm=llm)
    print(await react.run(user_msg="What is the ReAct pattern?"))

    print("\n== 3. Multi-agent with handoff + shared state ==")
    team = build_multi_agent()
    team_ctx = Context(team)
    print(await run_and_show_tools(
        team.run(user_msg="What is the latest LlamaIndex release?", ctx=team_ctx)
    ))
    print("Searches recorded in state:", (await team_ctx.store.get("state"))["searches"])


async def chat():
    agent = build_single_agent()
    ctx = Context(agent)
    print("Study assistant ready. Type 'quit' to exit.")
    while True:
        msg = input("\nYou: ").strip()
        if msg.lower() in {"quit", "exit"}:
            break
        print(f"\nAssistant: {await run_and_show_tools(agent.run(user_msg=msg, ctx=ctx))}")