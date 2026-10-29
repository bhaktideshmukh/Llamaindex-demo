from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from llama_index.utils.workflow import draw_all_possible_flows

from components import get_query_engine
from config import llm

MAX_WORDS = 30      # strict on purpose, so the loop is likely to trigger
MAX_ATTEMPTS = 3


class RetrieveEvent(Event):
    notes: str


class SummaryEvent(Event):
    summary: str


class TooLongEvent(Event):
    word_count: int


class SummaryWorkflow(Workflow):
    """retrieve -> summarize -> check -> (done | back to summarize)"""

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieveEvent:
        await ctx.store.set("attempts", 0)  # state shared by all steps
        response = await get_query_engine().aquery(f"Explain: {ev.topic}")
        return RetrieveEvent(notes=str(response))

    @step
    async def summarize(self, ctx: Context, ev: RetrieveEvent | TooLongEvent) -> SummaryEvent:
        if isinstance(ev, RetrieveEvent):
            await ctx.store.set("notes", ev.notes)
            hint = ""
        else:
            hint = f" Your last attempt had {ev.word_count} words, which is too long."
        notes = await ctx.store.get("notes")
        result = await llm.acomplete(f"Summarize in at most {MAX_WORDS} words.{hint}\n\n{notes}")
        return SummaryEvent(summary=str(result).strip())

    @step
    async def check(self, ctx: Context, ev: SummaryEvent) -> StopEvent | TooLongEvent:
        attempts = (await ctx.store.get("attempts")) + 1
        await ctx.store.set("attempts", attempts)
        words = len(ev.summary.split())
        print(f"Attempt {attempts}: {words} words")
        if words <= MAX_WORDS or attempts >= MAX_ATTEMPTS:
            return StopEvent(result=ev.summary)
        return TooLongEvent(word_count=words)  # loops back to summarize


async def demo():
    w = SummaryWorkflow(timeout=120)
    print(await w.run(topic="RAG"))
    draw_all_possible_flows(w, "flow.html")
    print("Workflow diagram saved to flow.html")