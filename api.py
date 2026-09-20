from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from semantic_search import SemanticSearchEngine
from qa import ExtractiveQAModel

app = FastAPI(
    title="Philosophy Semantic Search + Extractive QA",
    version="1.0.0",
)

HTML = """<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Philosophy QA — Roll No. 09</title>
<style>
body{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 18px;line-height:1.5}
input{width:100%;padding:14px;font-size:16px;box-sizing:border-box}
button{margin-top:12px;padding:12px 20px;font-size:16px;cursor:pointer}
.card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:12px 0}
.answer{font-size:20px;font-weight:600}
small{color:#666}
</style>
</head>
<body>
<h1>Philosophy Semantic Search + Extractive QA</h1>
<p><b>Experiment 7 | Roll No. 09 | Topic: Philosophy</b></p>
<p>Semantic search uses true sentence embeddings and cosine similarity.
The selected document is then passed to an extractive QA model.</p>
<input id="q" placeholder="Example: What does Stoicism teach about control?">
<button onclick="ask()">Search & Answer</button>
<div id="out"></div>
<script>
async function ask(){
  const q=document.getElementById("q").value.trim();
  if(!q){return;}
  document.getElementById("out").innerHTML="<p>Searching...</p>";
  const r=await fetch("/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q,top_k:3})});
  const d=await r.json();
  if(!r.ok){document.getElementById("out").innerHTML="<p>"+(d.detail||"Error")+"</p>";return;}
  let html="<h2>Semantic Search Results</h2>";
  d.results.forEach((x,i)=>html+=`<div class="card"><b>${i+1}. ${x.title}</b><br><small>Cosine similarity: ${x.score.toFixed(4)}</small></div>`);
  html+=`<h2>Extractive QA</h2><div class="card"><small>Selected: ${d.selected_document}</small><p class="answer">${d.answer}</p><small>QA confidence: ${d.qa_score.toFixed(4)}</small></div>`;
  document.getElementById("out").innerHTML=html;
}
</script>
</body>
</html>"""

class AskRequest(BaseModel):
    question: str = Field(min_length=2)
    top_k: int = Field(default=3, ge=1, le=5)

@lru_cache(maxsize=1)
def get_search_engine():
    return SemanticSearchEngine()

@lru_cache(maxsize=1)
def get_qa_model():
    return ExtractiveQAModel()

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.get("/health")
def health():
    return {"status": "ok", "roll_no": "09", "topic": "Philosophy"}

@app.post("/ask")
def ask(request: AskRequest):
    try:
        engine = get_search_engine()
        results = engine.search(request.question, request.top_k)
        best = results[0]

        qa = get_qa_model()
        answer = qa.answer(request.question, best["content"])

        return {
            "question": request.question,
            "selected_document": best["title"],
            "answer": answer["answer"],
            "qa_score": answer["score"],
            "results": [
                {"title": r["title"], "score": r["score"]}
                for r in results
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
