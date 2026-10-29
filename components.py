import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import embed_model, llm

def get_vector_store(reset: bool = False) -> ChromaVectorStore:
    """STORING: persistent Chroma collection on disk."""
    db = chromadb.PersistentClient(path="./chroma_db")
    if reset:
        try:
            db.delete_collection("notes")
        except Exception:
            pass
    return ChromaVectorStore(chroma_collection=db.get_or_create_collection("notes"))


async def ingest(data_dir: str = "./data"):
    """LOADING -> chunking -> embedding -> storing."""
    documents = SimpleDirectoryReader(input_dir=data_dir).load_data()
    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=256, chunk_overlap=20), embed_model],
        vector_store=get_vector_store(reset=True),
    )
    return await pipeline.arun(documents=documents)


def get_query_engine(response_mode: str = "compact"):
    """INDEXING + QUERYING. Modes: compact (default), refine, tree_summarize."""
    index = VectorStoreIndex.from_vector_store(get_vector_store(), embed_model=embed_model)
    return index.as_query_engine(llm=llm, response_mode=response_mode, similarity_top_k=3)


async def demo():
    nodes = await ingest()
    print(f"Ingested {len(nodes)} chunks\n")

    question = "What are the five stages of RAG?"
    for mode in ["compact", "refine", "tree_summarize"]:
        print(f"[{mode}] {await get_query_engine(mode).aquery(question)}\n")

    # EVALUATION (LLM-as-judge)
    response = await get_query_engine().aquery(question)
    faithful = await FaithfulnessEvaluator(llm=llm).aevaluate_response(
        query=question, response=response
    )
    relevant = await AnswerRelevancyEvaluator(llm=llm).aevaluate_response(
        query=question, response=response
    )
    correct = await CorrectnessEvaluator(llm=llm).aevaluate(
        query=question,
        response=str(response),
        reference="Loading, indexing, storing, querying and evaluation.",
    )
    print("Faithful to the notes:", faithful.passing)
    print("Relevant to the question:", relevant.passing)
    print("Correct vs reference:", correct.passing, f"(score {correct.score})")