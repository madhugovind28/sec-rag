import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .utils import ensure_dir, save_json, save_jsonl


def simple_tokenize(text: str) -> List[str]:
    return [tok for tok in text.lower().split() if tok.strip()]


def retrieval_text(chunk: Dict) -> str:
    header = " ".join(
        part
        for part in [
            chunk.get("company", ""),
            chunk.get("ticker", ""),
            chunk.get("form_type", ""),
            chunk.get("filing_date", ""),
            chunk.get("section_code", ""),
            chunk.get("section_title", ""),
        ]
        if part
    )
    return f"{header}\n{chunk['text']}"


def _latest_by_section(chunks: List[Dict], keywords: List[str], limit: int = 700) -> str:
    matches = []
    for chunk in chunks:
        title = (chunk.get("section_title") or "").lower()
        if any(k in title for k in keywords):
            matches.append(chunk)
    if not matches:
        return ""
    matches.sort(key=lambda c: (c.get("filing_date", ""), len(c.get("text", ""))), reverse=True)
    return matches[0].get("text", "")[:limit]


def _latest_by_form_and_section(chunks: List[Dict], form_type: str, keywords: List[str], limit: int = 700) -> str:
    matches = []
    for chunk in chunks:
        title = (chunk.get("section_title") or "").lower()
        form = (chunk.get("form_type") or "").upper()
        if form == form_type and any(k in title for k in keywords):
            matches.append(chunk)
    if not matches:
        return ""
    matches.sort(key=lambda c: (c.get("filing_date", ""), len(c.get("text", ""))), reverse=True)
    return matches[0].get("text", "")[:limit]


def _identity_fallback(chunks: List[Dict]) -> str:
    preferred = []
    for chunk in chunks:
        title = (chunk.get("section_title") or "").lower()
        if any(k in title for k in ["business", "management", "discussion", "analysis", "md&a"]):
            preferred.append(chunk)
    if not preferred:
        preferred = chunks[:3]
    preferred.sort(key=lambda c: (c.get("filing_date", ""), len(c.get("text", ""))), reverse=True)
    return "\n\n".join(c.get("text", "")[:450] for c in preferred[:2])


def build_company_profiles(chunks: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for chunk in chunks:
        ticker = (chunk.get("ticker") or "").upper()
        if ticker:
            grouped[ticker].append(chunk)

    profiles: List[Dict] = []
    for ticker, company_chunks in grouped.items():
        company_chunks.sort(key=lambda c: (c.get("filing_date", ""), len(c.get("text", ""))), reverse=True)
        company_name = company_chunks[0].get("company") or ticker
        latest_filing_date = company_chunks[0].get("filing_date", "")

        # Identity-heavy fields for dynamic company routing.
        business_10k = _latest_by_form_and_section(company_chunks, "10K", ["business"], limit=900)
        business_any = _latest_by_section(company_chunks, ["business"], limit=700)
        mda = _latest_by_section(company_chunks, ["md&a", "management", "discussion", "analysis"], limit=500)
        risk = _latest_by_section(company_chunks, ["risk factors", "market risk", "legal proceedings"], limit=300)
        fallback = _identity_fallback(company_chunks)

        identity_core = business_10k or business_any or mda or fallback
        secondary = business_any if business_any and business_any != identity_core else mda

        profile_text = "\n".join(
            part
            for part in [
                f"Company: {company_name}",
                f"Ticker: {ticker}",
                f"Latest filing date: {latest_filing_date}",
                # Repeat identity-bearing text so routing prefers what the company is.
                f"Company description: {identity_core}" if identity_core else "",
                f"Primary business: {identity_core}" if identity_core else "",
                f"Business detail: {secondary}" if secondary else "",
                f"Management discussion: {mda}" if mda else "",
                # Keep only a small amount of risk/legal text so generic regulation doesn't dominate routing.
                f"Limited risk context: {risk}" if risk else "",
            ]
            if part
        )

        profiles.append(
            {
                "ticker": ticker,
                "company": company_name,
                "latest_filing_date": latest_filing_date,
                "profile_text": profile_text,
            }
        )

    profiles.sort(key=lambda p: p["ticker"])
    return profiles


def _encode_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")


def _save_bm25(path: Path, texts: List[str]) -> None:
    tokenized_corpus = [simple_tokenize(t) for t in tqdm(texts, desc=f"Tokenizing for BM25 ({path.stem})")]
    bm25 = BM25Okapi(tokenized_corpus)
    with path.open("wb") as f:
        pickle.dump({"bm25": bm25, "tokenized_corpus": tokenized_corpus}, f)


def build_indices(chunks: List[Dict], out_dir: str, embed_model_name: str) -> None:
    out = ensure_dir(Path(out_dir))
    model = SentenceTransformer(embed_model_name)

    # Chunk-level index
    chunk_texts = [retrieval_text(c) for c in chunks]
    chunk_embeddings = _encode_texts(model, chunk_texts)

    chunk_index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
    chunk_index.add(chunk_embeddings)
    faiss.write_index(chunk_index, str(out / "faiss.index"))

    _save_bm25(out / "bm25.pkl", chunk_texts)
    np.save(out / "embeddings.npy", chunk_embeddings)
    save_jsonl(out / "chunks.jsonl", chunks)

    # Company-level routing index
    company_profiles = build_company_profiles(chunks)
    company_texts = [p["profile_text"] for p in company_profiles]
    company_embeddings = _encode_texts(model, company_texts)

    company_index = faiss.IndexFlatIP(company_embeddings.shape[1])
    company_index.add(company_embeddings)
    faiss.write_index(company_index, str(out / "company_faiss.index"))

    _save_bm25(out / "company_bm25.pkl", company_texts)
    np.save(out / "company_embeddings.npy", company_embeddings)
    save_jsonl(out / "company_profiles.jsonl", company_profiles)

    save_json(
        out / "index_meta.json",
        {
            "num_chunks": len(chunks),
            "num_companies": len(company_profiles),
            "embedding_model": embed_model_name,
            "vector_dim": int(chunk_embeddings.shape[1]),
        },
    )