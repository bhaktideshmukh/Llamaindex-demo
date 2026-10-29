from datetime import date

from ddgs import DDGS
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.tools.tool_spec.load_and_search import LoadAndSearchToolSpec

from components import get_query_engine


# 1. FunctionTool: any Python function. Name, type hints and docstring are what the LLM sees.
def web_search(query: str) -> str:
    """Search the web for current information on a topic and return the top results."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']}\n{r['body']}\n{r['href']}" for r in results)


def get_today() -> str:
    """Return today's date in YYYY-MM-DD format."""
    return date.today().isoformat()


web_search_tool = FunctionTool.from_defaults(web_search)
today_tool = FunctionTool.from_defaults(
    get_today, name="get_today", description="Returns today's date."
)


# 2. QueryEngineTool: wraps the RAG query engine so an agent can use it (agentic RAG)
def make_notes_tool() -> QueryEngineTool:
    return QueryEngineTool.from_defaults(
        get_query_engine(),
        name="study_notes",
        description="Answers questions about the user's study notes on AI agents and RAG.",
    )


# 3. ToolSpec: a bundle of related tools
class TextToolSpec(BaseToolSpec):
    spec_functions = ["count_words", "reading_time_minutes"]

    def count_words(self, text: str) -> int:
        """Count the number of words in a piece of text."""
        return len(text.split())

    def reading_time_minutes(self, text: str) -> float:
        """Estimate how many minutes a text takes to read at 200 words per minute."""
        return round(len(text.split()) / 200, 2)


text_tools = TextToolSpec().to_tool_list()


# 4. Utility tool: splits one tool into a LOAD tool and a SEARCH tool, so large
#    outputs get indexed instead of flooding the LLM's context window
def make_load_and_search_tools():
    return LoadAndSearchToolSpec.from_defaults(web_search_tool).to_tool_list()


async def demo():
    print("== FunctionTool ==")
    print(today_tool.metadata.name, "|", today_tool.metadata.description)
    print("Direct call:", today_tool.call().content)

    print("\n== QueryEngineTool ==")
    out = await make_notes_tool().acall(input="What happens to documents before they are stored?")
    print(out.content)

    print("\n== ToolSpec ==")
    for t in text_tools:
        print(t.metadata.name, "|", t.metadata.description)
    print("Direct call:", text_tools[0].call(text="one two three").content)

    print("\n== Utility tool (load + search) ==")
    load_tool, search_tool = make_load_and_search_tools()
    print(load_tool.call(query="LlamaIndex agents").content)
    print(search_tool.call(query="What is LlamaIndex?").content)