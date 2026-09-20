"""FastAPI service. Run: uvicorn hybrid_rag.api:app --reload"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .pipeline import HybridRAG

app = FastAPI(
    title="Enterprise Hybrid RAG",
    description="Vector RAG + Graph RAG with GPT, demonstrated on a supermarket chain.",
    version=__version__,
)


@lru_cache(maxsize=1)
def get_rag() -> HybridRAG:
    return HybridRAG().load()


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, examples=["FreshDairy milk is recalled. Which stores and managers are affected?"]
    )
    roles: list[str] = Field(default_factory=lambda: ["staff"], examples=[["manager"]])
    route: Literal["auto", "vector", "graph", "hybrid"] = "auto"


class IngestRequest(BaseModel):
    extract_graph: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        rag = get_rag()
        return rag.query(req.question, roles=req.roles, route=None if req.route == "auto" else req.route).to_dict()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    return get_rag().ingest(extract_graph=req.extract_graph)


@app.get("/graph/stats")
def graph_stats() -> dict:
    return get_rag().graph.stats()
