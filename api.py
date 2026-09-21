from functools import lru_cache
from html import escape

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from qa import ExtractiveQAModel
from semantic_search import SemanticSearchEngine

app = FastAPI(title="Philosophy Semantic Search + Extractive QA", version="2.0.0")

HTML = """<!doctype html>
<html><head><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Philosophy QA</title>
<style>
body{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 18px;line-height:1.5}
input{width:100%;padding:14px;font-size:16px;box-sizing:border-box}button{margin-top:12px;padding:12px 20px;font-size:16px;cursor:pointer}
.card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:12px 0}.answer{font-size:20px;font-weight:600}small{color:#666}.error{color:#a00}
</style></head><body>
<h1>Philosophy Semantic Search + Extractive QA</h1>
<p><b>Experiment 7 | Roll No. 09 | Topic: Philosophy</b></p>
<p>Ask a question about one of the available philosophy topics.</p>
<input id="q" placeholder="Example: What does Stoicism teach about control?" onkeydown="if(event.key==='Enter')ask()">
<button id="button" onclick="ask()">Search &amp; Answer</button><div id="out"></div>
<script>
async function ask(){
 const q=document.getElementById('q').value.trim(), out=document.getElementById('out'), button=document.getElementById('button');
 if(!q){out.innerHTML='<p class="error">Please enter a question.</p>';return;}
 button.disabled=true; out.innerHTML='<p>Searching...</p>';
 try { const response=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,top_k:3})});
 const data=await response.json(); if(!response.ok) throw new Error(data.detail||'Request failed');
 let html='<h2>Semantic Search Results</h2>'; data.results.forEach((x,i)=>{html+=`<div class="card"><b>${i+1}. ${x.title}</b><br><small>Cosine similarity: ${x.score.toFixed(4)}</small></div>`});
 html+=`<h2>Answer</h2><div class="card"><small>Selected: ${data.selected_document}</small><p class="answer">${data.answer}</p><small>Confidence: ${data.qa_score.toFixed(4)}</small></div>`; out.innerHTML=html;
 } catch(error){out.innerHTML='<p class="error">'+error.message+'</p>'} finally{button.disabled=false}
}
</script></body></html>"""


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


@lru_cache(maxsize=1)
def get_search_engine() -> SemanticSearchEngine:
    return SemanticSearchEngine()


@lru_cache(maxsize=1)
def get_qa_model() -> ExtractiveQAModel:
    return ExtractiveQAModel()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HTML


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "qa_mode": "local", "hf_token_required": False}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    try:
        results = get_search_engine().search(request.question, request.top_k)
        if not results:
            raise ValueError("No philosophy documents are available.")
        best = results[0]
        answer = get_qa_model().answer(request.question, best["content"])
        return {
            "question": request.question,
            "selected_document": best["title"],
            "answer": answer["answer"],
            "qa_score": answer["score"],
            "results": [{"title": r["title"], "score": r["score"]} for r in results],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
