import os
import pickle
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import (
    BM25_TOP_K,
    COMPANY_ROUTE_TOP_K,
    DENSE_TOP_K,
    FINAL_TOP_K,
    MAX_CHUNKS_PER_COMPANY,
    MAX_CHUNKS_PER_SECTION,
    RRF_K,
)
from .indexing import simple_tokenize
from .utils import load_json, load_jsonl


STOPWORDS = {
    "inc", "corp", "corporation", "company", "co", "holdings", "holding", "group",
    "plc", "ltd", "limited", "the", "and", "of", "class", "a", "common", "stock"
}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

ROUTE_QUERY_STOPWORDS = {
    "what", "which", "who", "how", "when", "where", "why", "are", "is", "do", "does",
    "did", "have", "has", "had", "the", "a", "an", "and", "or", "of", "to", "in", "on",
    "for", "with", "their", "them", "they", "face", "facing", "address", "addressing",
    "major", "companies", "company", "risks", "risk", "regulatory", "legal", "compliance",
    "latest", "recent", "last", "past", "years", "year", "quarters", "quarter", "over", "time",
    "changed", "change", "outlook", "growth", "revenue", "guidance", "compare", "comparison",
}

IDENTITY_TERM_EXPANSIONS = {
    "pharmaceutical": ["pharmaceutical", "pharmaceuticals", "pharma", "biopharma", "biopharmaceutical", "drug", "drugs", "medicine", "medicines", "therapeutic", "therapeutics", "biologic", "biologics", "vaccine", "vaccines"],
    "pharma": ["pharmaceutical", "pharmaceuticals", "pharma", "biopharma", "biopharmaceutical", "drug", "drugs", "medicine", "medicines", "therapeutic", "therapeutics", "biologic", "biologics", "vaccine", "vaccines"],
    "bank": ["bank", "banks", "banking", "financial", "finance", "deposits", "loans", "lending", "credit", "payments", "capital markets"],
    "banks": ["bank", "banks", "banking", "financial", "finance", "deposits", "loans", "lending", "credit", "payments", "capital markets"],
    "semiconductor": ["semiconductor", "semiconductors", "chip", "chips", "gpu", "cpu", "foundry", "wafer"],
    "retail": ["retail", "retailer", "retailers", "consumer", "stores", "e-commerce", "commerce"],
    "airline": ["airline", "airlines", "aviation", "air travel", "passenger", "fleet"],
}



class HybridRetriever:
    def __init__(self, index_dir: str):
        index_dir = Path(index_dir)
        meta = load_json(index_dir / "index_meta.json")
        self.embedding_model_name = meta["embedding_model"]
        self.model = SentenceTransformer(self.embedding_model_name)

        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.chunks = load_jsonl(index_dir / "chunks.jsonl")
        with (index_dir / "bm25.pkl").open("rb") as f:
            bm25_state = pickle.load(f)
        self.bm25 = bm25_state["bm25"]

        # Optional company-level routing index.
        self.company_profiles = []
        self.company_index = None
        self.company_bm25 = None
        company_profiles_path = index_dir / "company_profiles.jsonl"
        company_index_path = index_dir / "company_faiss.index"
        company_bm25_path = index_dir / "company_bm25.pkl"
        if company_profiles_path.exists() and company_index_path.exists() and company_bm25_path.exists():
            self.company_profiles = load_jsonl(company_profiles_path)
            self.company_index = faiss.read_index(str(company_index_path))
            with company_bm25_path.open("rb") as f:
                state = pickle.load(f)
            self.company_bm25 = state["bm25"]

        self.company_aliases = self._build_company_aliases()
        env_today = os.getenv("RAG_TODAY", "").strip()
        self.today = self._parse_filing_date(env_today) or date.today()

    def _parse_filing_date(self, value: str) -> Optional[date]:
        if not value:
            return None
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except Exception:
            return None

    def _normalize_company_name(self, name: str) -> str:
        name = name.lower().replace('&', ' and ')
        name = re.sub(r'[^a-z0-9\s]', ' ', name)
        parts = [p for p in name.split() if p and p not in STOPWORDS]
        return ' '.join(parts)

    def _build_company_aliases(self) -> Dict[str, Set[str]]:
        aliases: Dict[str, Set[str]] = defaultdict(set)
        source_rows = self.company_profiles if self.company_profiles else self.chunks
        for row in source_rows:
            ticker = (row.get("ticker") or "").strip().upper()
            company = (row.get("company") or "").strip()
            if not ticker:
                continue
            aliases[ticker].add(ticker.lower())
            if company:
                aliases[ticker].add(company.lower())
                normalized = self._normalize_company_name(company)
                if normalized:
                    aliases[ticker].add(normalized)
                    parts = normalized.split()
                    if parts:
                        aliases[ticker].add(parts[0])
                    if len(parts) >= 2:
                        aliases[ticker].add(' '.join(parts[:2]))
        return aliases

    def _mentioned_tickers(self, query: str) -> Set[str]:
        query_lower = query.lower()
        normalized_query = self._normalize_company_name(query)
        mentioned: Set[str] = set()
        for ticker, names in self.company_aliases.items():
            for name in sorted(names, key=len, reverse=True):
                if not name:
                    continue
                if ' ' in name:
                    if name in query_lower or (normalized_query and name in normalized_query):
                        mentioned.add(ticker)
                        break
                else:
                    pattern = rf"\b{re.escape(name)}\b"
                    if re.search(pattern, query_lower) or (normalized_query and re.search(pattern, normalized_query)):
                        mentioned.add(ticker)
                        break
        return mentioned

    def _query_mode(self, query: str, mentioned: Set[str]) -> str:
        query_lower = query.lower()
        comparison_terms = [
            "compare", "comparison", "versus", "vs", "differ", "difference",
            "similar", "across", "how do they compare",
        ]
        if len(mentioned) == 1:
            return "single_company"
        if len(mentioned) > 1 or any(term in query_lower for term in comparison_terms):
            return "comparison"
        return "broad"

    def _company_route_terms(self, query: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z-]+", query.lower())
        informative = []
        for tok in tokens:
            if tok in ROUTE_QUERY_STOPWORDS:
                continue
            if len(tok) <= 2:
                continue
            informative.append(tok)
        seen = set()
        ordered = []
        for tok in informative:
            if tok not in seen:
                ordered.append(tok)
                seen.add(tok)
        return ordered[:6]

    def _expanded_identity_terms(self, route_terms: List[str]) -> List[str]:
        expanded: List[str] = []
        for term in route_terms:
            expanded.append(term)
            if term.endswith('s') and len(term) > 4:
                expanded.append(term[:-1])
            if term in IDENTITY_TERM_EXPANSIONS:
                expanded.extend(IDENTITY_TERM_EXPANSIONS[term])
            if term.startswith('pharma'):
                expanded.extend(IDENTITY_TERM_EXPANSIONS['pharmaceutical'])
            if term.startswith('bank'):
                expanded.extend(IDENTITY_TERM_EXPANSIONS['bank'])
        seen = set()
        out = []
        for term in expanded:
            t = term.lower().strip()
            if t and t not in seen:
                out.append(t)
                seen.add(t)
        return out

    def _profile_identity_text(self, profile: Dict) -> str:
        text = profile.get("profile_text", "") or ""
        # Lean heavily on the beginning of the profile, which is business-description-centric.
        return text[:900].lower()

    def _identity_match_count(self, profile: Dict, identity_terms: List[str]) -> int:
        if not identity_terms:
            return 0
        profile_text = self._profile_identity_text(profile)
        return sum(1 for term in identity_terms if term in profile_text)

    def _temporal_preferences(self, query: str) -> Dict[str, object]:
        query_lower = query.lower()
        prefs = {"is_temporal": False, "prefer_recent": False, "lookback_days": None}

        if any(term in query_lower for term in [
            "changed", "over time", "trend", "latest", "recent", "currently", "now",
            "outlook", "guidance", "last year", "last quarter", "past year", "past quarter",
            "last two years", "past two years", "over the last two years"
        ]):
            prefs["is_temporal"] = True
            prefs["prefer_recent"] = True

        m = re.search(r"(?:last|past|over the last)\s+(\d+|one|two|three|four|five)\s+(year|years|quarter|quarters|month|months)", query_lower)
        if m:
            raw_num = m.group(1)
            unit = m.group(2)
            n = int(raw_num) if raw_num.isdigit() else NUMBER_WORDS.get(raw_num, 1)
            if "year" in unit:
                prefs["lookback_days"] = 365 * n
            elif "quarter" in unit:
                prefs["lookback_days"] = 90 * n
            elif "month" in unit:
                prefs["lookback_days"] = 30 * n
            prefs["is_temporal"] = True
            prefs["prefer_recent"] = True
        return prefs

    def _window_start_date(self, temporal_prefs: Dict[str, object]) -> Optional[date]:
        lookback_days = temporal_prefs.get("lookback_days")
        if not lookback_days:
            return None
        return self.today - timedelta(days=int(lookback_days))

    def _recency_adjustment(self, filing_date: Optional[date], temporal_prefs: Dict[str, object]) -> float:
        if not temporal_prefs.get("prefer_recent") or not filing_date:
            return 0.0

        age_days = max(0, (self.today - filing_date).days)
        lookback_days = temporal_prefs.get("lookback_days")
        if lookback_days:
            if age_days <= lookback_days:
                freshness = 1.0 - (age_days / max(lookback_days, 1))
                return 0.08 + 0.04 * freshness
            return -0.08
        years_old = age_days / 365.0
        return max(-0.03, 0.08 - 0.02 * years_old)

    def dense_search(self, query: str, top_k: int = DENSE_TOP_K) -> List[int]:
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        _, ids = self.index.search(q, top_k)
        return [int(i) for i in ids[0] if i != -1]

    def bm25_search(self, query: str, top_k: int = BM25_TOP_K) -> List[int]:
        scores = self.bm25.get_scores(simple_tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return [int(i) for i in order]

    def _route_companies(self, query: str, top_k: int = COMPANY_ROUTE_TOP_K) -> List[str]:
        if not self.company_index or not self.company_profiles or not self.company_bm25:
            return []

        route_terms = self._company_route_terms(query)
        identity_terms = self._expanded_identity_terms(route_terms)
        company_query = " ".join(identity_terms[:8]) if identity_terms else query

        q = self.model.encode([company_query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        dense_k = min(max(top_k * 3, 10), len(self.company_profiles))
        dense_scores, dense_ids = self.company_index.search(q, dense_k)
        dense_scores = dense_scores[0]
        dense_ids = [int(i) for i in dense_ids[0] if i != -1]

        bm25_scores = self.company_bm25.get_scores(simple_tokenize(company_query))
        bm25_order = np.argsort(bm25_scores)[::-1][:dense_k]
        bm25_ids = [int(i) for i in bm25_order]

        fused = defaultdict(float)
        for rank, idx in enumerate(dense_ids):
            fused[idx] += 1.0 / (RRF_K + rank + 1)
            fused[idx] += 0.05 * float(dense_scores[rank])
        for rank, idx in enumerate(bm25_ids):
            fused[idx] += 1.0 / (RRF_K + rank + 1)
            fused[idx] += 0.02 * float(bm25_scores[idx])

        # Identity-aware lexical reranking: favor profiles whose business-description text
        # explicitly contains the query's informative category words.
        if identity_terms:
            for idx, profile in enumerate(self.company_profiles):
                match_count = self._identity_match_count(profile, identity_terms)
                if match_count:
                    fused[idx] += 0.45 * match_count
                else:
                    fused[idx] -= 0.35

        ranked = [idx for idx, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]
        tickers = []
        seen = set()
        for idx in ranked:
            profile = self.company_profiles[idx]
            ticker = (profile.get("ticker") or "").upper()
            if not ticker or ticker in seen:
                continue
            if identity_terms:
                if self._identity_match_count(profile, identity_terms) == 0:
                    continue
            tickers.append(ticker)
            seen.add(ticker)
            if len(tickers) >= top_k:
                break

        if len(tickers) < min(top_k, 2):
            for idx in ranked:
                ticker = (self.company_profiles[idx].get("ticker") or "").upper()
                if ticker and ticker not in seen:
                    tickers.append(ticker)
                    seen.add(ticker)
                if len(tickers) >= top_k:
                    break
        return tickers

    def _company_fallback_ids(
        self,
        ticker: str,
        bm25_scores: np.ndarray,
        query: str,
        temporal_prefs: Dict[str, object],
        window_start: Optional[date],
        limit: int = 5,
    ) -> List[int]:
        query_lower = query.lower()
        candidates: List[Tuple[int, float]] = []
        for idx, chunk in enumerate(self.chunks):
            if (chunk.get("ticker") or "").upper() != ticker:
                continue

            filing_date = self._parse_filing_date(chunk.get("filing_date", ""))
            if window_start and filing_date and filing_date < window_start:
                continue

            score = float(bm25_scores[idx]) + self._recency_adjustment(filing_date, temporal_prefs)
            section_title = (chunk.get("section_title") or "").lower()
            text_head = (chunk.get("text") or "")[:500].lower()

            if any(term in query_lower for term in ["risk", "regulatory", "legal", "compliance"]):
                if any(term in section_title for term in ["risk", "legal", "proceeding"]):
                    score += 1.0
                if any(term in text_head for term in ["regulatory", "fda", "approval", "compliance", "warning", "enforcement"]):
                    score += 0.6
            if any(term in query_lower for term in ["revenue", "growth", "outlook", "guidance", "forecast"]):
                if any(term in section_title for term in ["management", "discussion", "analysis", "md&a"]):
                    score += 1.4
                if "financial statements" in section_title and any(term in text_head for term in ["revenue", "segment", "data center", "gaming", "automotive", "networking"]):
                    score += 1.0
                if any(term in text_head for term in ["revenue", "growth", "outlook", "guidance", "demand", "segment", "data center", "gaming", "automotive", "networking"]):
                    score += 0.8
                if "risk factors" in section_title:
                    score -= 1.1
                if "business" in section_title:
                    score -= 0.15
            else:
                if "risk factors" in section_title:
                    score += 0.4
                if "business" in section_title:
                    score += 0.2

            candidates.append((idx, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in candidates[:limit]]

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> List[Dict]:
        dense_ids = self.dense_search(query)
        bm25_ids = self.bm25_search(query)
        bm25_scores = self.bm25.get_scores(simple_tokenize(query))
        fused = defaultdict(float)
        query_lower = query.lower()
        mentioned_tickers = self._mentioned_tickers(query)
        query_mode = self._query_mode(query, mentioned_tickers)
        temporal_prefs = self._temporal_preferences(query)
        window_start = self._window_start_date(temporal_prefs)

        for rank, idx in enumerate(dense_ids):
            fused[idx] += 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm25_ids):
            fused[idx] += 1.0 / (RRF_K + rank + 1)

        # Decide which companies are allowed in this query.
        routed_tickers: List[str] = []
        if not mentioned_tickers:
            routed_tickers = self._route_companies(query, top_k=COMPANY_ROUTE_TOP_K)
        allowed_tickers: Set[str] = set(mentioned_tickers) if mentioned_tickers else set(routed_tickers)

        for idx in list(fused.keys()):
            chunk = self.chunks[idx]
            filing_date = self._parse_filing_date(chunk.get("filing_date", ""))
            if window_start and filing_date and filing_date < window_start:
                del fused[idx]
                continue

            ticker = (chunk.get("ticker") or "").upper()
            company = (chunk.get("company") or "").lower()
            if ticker in mentioned_tickers or (company and company in query_lower):
                fused[idx] += 0.10
            if allowed_tickers and ticker in allowed_tickers:
                fused[idx] += 0.08

            fused[idx] += self._recency_adjustment(filing_date, temporal_prefs)

            section_title = (chunk.get("section_title") or "").lower()
            text_head = (chunk.get("text") or "")[:500].lower()
            if any(term in query_lower for term in ["revenue", "growth", "outlook", "guidance", "forecast"]):
                if any(term in section_title for term in ["management", "discussion", "analysis", "md&a"]):
                    fused[idx] += 0.35
                if "financial statements" in section_title and any(term in text_head for term in ["revenue", "segment", "data center", "gaming", "automotive", "networking"]):
                    fused[idx] += 0.24
                if any(term in text_head for term in ["revenue", "growth", "outlook", "guidance", "demand", "data center", "segment", "gaming", "automotive", "networking"]):
                    fused[idx] += 0.16
                if "risk factors" in section_title:
                    fused[idx] -= 0.45
                if "business" in section_title:
                    fused[idx] -= 0.06
            if any(term in query_lower for term in ["regulatory", "compliance", "legal", "approval", "drug", "pharma", "pharmaceutical", "bank"]):
                if any(term in section_title for term in ["risk", "legal", "proceeding", "business"]):
                    fused[idx] += 0.04
                if any(term in text_head for term in ["regulatory", "fda", "approval", "compliance", "banking", "capital", "reimbursement"]):
                    fused[idx] += 0.03

        ranked_ids = [idx for idx, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

        # Restrict to named companies or dynamically routed companies when available.
        filtered_ranked_ids = [
            idx for idx in ranked_ids
            if not allowed_tickers or (self.chunks[idx].get("ticker") or "").upper() in allowed_tickers
        ]

        # Ensure each allowed company has a fallback pool inside the final candidate set.
        candidate_ids = list(filtered_ranked_ids)
        if allowed_tickers:
            for ticker in sorted(allowed_tickers):
                candidate_ids.extend(self._company_fallback_ids(ticker, bm25_scores, query, temporal_prefs, window_start, limit=8))
            deduped = []
            seen = set()
            for idx in candidate_ids:
                if idx not in seen:
                    deduped.append(idx)
                    seen.add(idx)
            candidate_ids = deduped

        if query_mode == "single_company":
            company_cap = top_k
            section_cap = max(MAX_CHUNKS_PER_SECTION, 3)
        else:
            company_cap = MAX_CHUNKS_PER_COMPANY
            section_cap = MAX_CHUNKS_PER_SECTION

        picked: List[Dict] = []
        picked_ids = set()
        company_counts = defaultdict(int)
        section_counts = defaultdict(int)

        def try_add(idx: int, ignore_company_cap: bool = False) -> bool:
            if idx in picked_ids:
                return False
            chunk = dict(self.chunks[idx])
            company_key = (chunk.get("ticker") or "").upper()
            section_key = (company_key, chunk.get("section_code", ""))
            if allowed_tickers and company_key not in allowed_tickers:
                return False
            if not ignore_company_cap and company_counts[company_key] >= company_cap:
                return False
            if section_counts[section_key] >= section_cap:
                return False
            picked.append(chunk)
            picked_ids.add(idx)
            company_counts[company_key] += 1
            section_counts[section_key] += 1
            return True

        # Seed one chunk per explicit or dynamically routed company first.
        if allowed_tickers and len(allowed_tickers) > 1:
            for ticker in list(allowed_tickers)[:top_k]:
                for idx in candidate_ids:
                    if (self.chunks[idx].get("ticker") or "").upper() == ticker:
                        if try_add(idx, ignore_company_cap=True):
                            break

        for idx in candidate_ids:
            try_add(idx)
            if len(picked) >= top_k:
                break

        return picked