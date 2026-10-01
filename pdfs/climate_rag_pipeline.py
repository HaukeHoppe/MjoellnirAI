"""
title: Climate PDF RAG (test pipeline)
author: Mjoelnir AI
date: 2026-09-22
version: 0.1
license: MIT
description: Smoke-test Open WebUI pipeline over a single static PDF (Understanding_Climate_Change.pdf). Validates the retrieve -> ground -> cite -> stream plumbing before porting the same skeleton to the capital-markets pipeline in RAG_Techniques/PROMPT_openwebui_pipeline.md. Not the final architecture: no concept/opinion split, no fusion/reranking, and the grounding check is a plain similarity-score threshold rather than the full reliable_rag logic.
requirements: langchain-community,langchain-openai,langchain-core,faiss-cpu,openai,pydantic
"""

from typing import Generator, Iterator, List, Union

from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class Pipeline:
    class Valves(BaseModel):
        INDEX_DIR: str = "/data/faiss_climate_index"
        # Must match the embedding model encode_pdf() used at ingestion time
        # (helper_functions.encode_pdf calls OpenAIEmbeddings() with no model
        # arg, which defaults to text-embedding-ada-002). A mismatch here
        # compares vectors from two different embedding spaces and produces
        # near-random distances.
        EMBEDDING_MODEL: str = "text-embedding-ada-002"
        LLM_MODEL: str = "gpt-4o-mini"
        TOP_K: int = 4
        # FAISS similarity_search_with_score returns L2 distance (lower = closer).
        # Calibrated from a handful of manual queries against this PDF's index:
        # relevant hits landed ~0.22-0.38, an unrelated/nonsense query ~0.55+.
        # Still a rough heuristic, not a substitute for the real reliable_rag
        # grounding logic - re-tune if you see false positives/negatives.
        MAX_GROUNDING_DISTANCE: float = 0.45

    def __init__(self):
        self.name = "Climate PDF RAG (test)"
        self.valves = self.Valves()
        self.vectorstore = None
        self.llm = None

    async def on_startup(self):
        # Loads the index built by ingest_climate_pdf.py. Does NOT run ingestion here.
        embeddings = OpenAIEmbeddings(model=self.valves.EMBEDDING_MODEL)
        self.vectorstore = FAISS.load_local(
            self.valves.INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        self.llm = ChatOpenAI(model=self.valves.LLM_MODEL, temperature=0, streaming=True)

    async def on_shutdown(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        results = self.vectorstore.similarity_search_with_score(
            user_message, k=self.valves.TOP_K
        )

        grounded = [
            (doc, score)
            for doc, score in results
            if score <= self.valves.MAX_GROUNDING_DISTANCE
        ]

        if not grounded:
            yield "I don't have a grounded answer for that in the indexed document."
            return

        context = "\n\n".join(doc.page_content for doc, _ in grounded)
        pages = sorted(
            {
                doc.metadata["page"]
                for doc, _ in grounded
                if doc.metadata.get("page") is not None
            }
        )

        if not pages:
            # No traceable page metadata on any grounded chunk - treat as ungrounded
            # rather than citing vaguely (same rule as the real pipeline's
            # citation requirement).
            yield "I don't have a grounded answer for that in the indexed document."
            return

        prompt = (
            "Answer the question using ONLY the context below. "
            "If the context doesn't support an answer, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {user_message}"
        )

        page_refs = ", ".join(f"p.{p + 1}" for p in pages)

        # .stream() yields tokens as OpenAI produces them, instead of
        # .invoke() which blocks until the full completion is done and
        # hands back one big string - that's what made answers appear
        # "printed all at once" before.
        for chunk in self.llm.stream(prompt):
            if chunk.content:
                yield chunk.content

        yield f"\n\nSource: Understanding_Climate_Change.pdf ({page_refs})"
