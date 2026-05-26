"""Mnemosyne HTTP service — lead sessions call this at boot and on exit."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mnemosyne import distiller, querier, store


app = FastAPI(title="mnemosyne", version="0.1.0")


class QueryRequest(BaseModel):
    domain: str
    question: str


class QueryResponse(BaseModel):
    domain: str
    question: str
    answer: str
    training_pairs_available: int


class DistillRequest(BaseModel):
    domain: str
    transcript: str
    session_slug: str = ""


class DistillResponse(BaseModel):
    domain: str
    pairs_extracted: int
    total_pairs: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "domains": store.domains()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    pairs = store.load(req.domain)
    answer = querier.query(req.domain, req.question)
    return QueryResponse(
        domain=req.domain,
        question=req.question,
        answer=answer,
        training_pairs_available=len(pairs),
    )


@app.post("/distill", response_model=DistillResponse)
def distill(req: DistillRequest) -> DistillResponse:
    pairs = distiller.distill(req.transcript, req.domain, req.session_slug)
    total = len(store.load(req.domain))
    return DistillResponse(
        domain=req.domain,
        pairs_extracted=len(pairs),
        total_pairs=total,
    )


@app.get("/domains")
def domains() -> dict:
    result = {}
    for d in store.domains():
        result[d] = len(store.load(d))
    return result
