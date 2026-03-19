# SEC Filings RAG Demo

This repo builds a **minimal retrieval-augmented generation (RAG)** demo over SEC EDGAR filings and answers a business question with **one final LLM API call** by combining metadata-aware chunking, FAISS dense retrieval, BM25 lexical retrieval, temporal filtering, and company-aware routing, while keeping the pipeline lightweight and reproducible for demo use.

The indexing, chunking, and retrieval happen ahead of time. At question time, the system:
1. retrieves the most relevant filing chunks,
2. injects them into a prompt,
3. makes one final local API call to Ollama.

---

## Why this implementation

I optimized for four things:
- **minimal code** that is easy to explain live,
- **fully free / local** runtime,
- **good enough retrieval quality** for long 10-K / 10-Q text,
- **clear grounding** with citations.

I deliberately did **not** add agent frameworks, hosted vector DBs, or multi-step tool calling because they add complexity without helping the core requirement.

---


## Setup

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3) Install Ollama
Install Ollama for your OS, then pull a model.

Recommended default:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:4b
```

You can swap in another local model if you prefer.

### 4) Put the corpus zip in `data/raw/`
Example:
```bash
cp /path/to/edgar_corpus.zip data/raw/
```

---

## Build the index

```bash
python scripts/build_index.py \
  --corpus data/raw/edgar_corpus.zip \
  --out data/index
```

This will:
- extract the zip into `data/.cache/`
- read all filings
- chunk them
- build FAISS + BM25 indices
- save chunk metadata into `data/index/`

---

## Ask questions in the warm CLI

Use the persistent CLI so the embedding model and retrieval indices are loaded once:

```bash
python scripts/chat.py --index-dir data/index --model gemma3:4b --chunks
```

Then type questions directly, for example:

```text
Question> What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?
```

---

## Repo layout

```text
sec-rag-demo/
├── README.md
├── requirements.txt
├── prompts/
│   ├── final_prompt.txt
│   └── prompt_iterations.md
├── evaluation/
│   └── eval_notes.md
├── scripts/
│   ├── build_index.py
│   └── chat.py
└── src/
    ├── config.py
    ├── ingest.py
    ├── chunking.py
    ├── indexing.py
    ├── retrieval.py
    ├── prompting.py
    ├── llm.py
    └── utils.py
```

---

## If I had more time
- use full sec parser
- add a local reranker
- tighten section parsing
- add company-name normalization
- add an evaluation set
- reduce latency via optimal indexing/ranking
- experiment with different chunk sizes, overlap sizes, and k values