from pathlib import Path
from typing import Dict, List

TEMPORAL_WORDS = (
    "last ",
    "past ",
    "over time",
    "changed",
    "change",
    "trend",
    "outlook",
    "growth",
    "guidance",
    "sequentially",
    "year-over-year",
    "yoy",
)

CATEGORY_HINT_WORDS = (
    "companies",
    "industry",
    "sector",
    "pharmaceutical",
    "pharma",
    "drugmakers",
    "banks",
    "banking",
    "semiconductor",
    "retailers",
    "insurers",
    "healthcare",
)


def citation_label(chunk: Dict) -> str:
    company = (chunk.get("company") or chunk.get("ticker") or "Unknown Company").strip()
    ticker = (chunk.get("ticker") or "").strip()
    section = chunk.get("section_code") or "Unknown"
    title = chunk.get("section_title") or "Unknown Section"
    company_label = f"{company} ({ticker})" if ticker else company
    return f"[{company_label} | {chunk['form_type']} | {chunk['filing_date']} | Item {section} {title}]"


def format_context(chunks: List[Dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Context {i}: {citation_label(chunk)}",
                    chunk["text"],
                ]
            )
        )
    return "\n\n".join(blocks)


def _company_labels(chunks: List[Dict]) -> List[str]:
    labels = []
    seen = set()
    for c in chunks:
        company = (c.get("company") or c.get("ticker") or "Unknown Company").strip()
        ticker = (c.get("ticker") or "").strip()
        label = f"{company} ({ticker})" if ticker else company
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _is_single_company_temporal(question: str, company_labels: List[str]) -> bool:
    if len(company_labels) != 1:
        return False
    q = question.lower()
    return any(term in q for term in TEMPORAL_WORDS)


def _is_category_query(question: str, company_labels: List[str]) -> bool:
    q = question.lower()
    if len(company_labels) > 1 and any(term in q for term in CATEGORY_HINT_WORDS):
        return True
    return any(term in q for term in CATEGORY_HINT_WORDS)


def _single_company_temporal_prefix(company_label: str) -> str:
    return (
        "SINGLE-COMPANY TEMPORAL ANALYSIS MODE\n"
        f"Primary company in retrieved context: {company_label}\n"
        "Use the exact company label above when naming the company.\n"
        "Focus on chronology, revenue/growth/outlook changes, and what became stronger or weaker over time.\n"
        "Synthesize across filings rather than repeating every metric.\n"
        "Do not mention any company that is not the primary company above.\n"
        "If the retrieved context does not cover part of the requested time window, say so explicitly in Gaps / uncertainty.\n"
        "Avoid saying 'Context 1' or 'Context 2' in the answer; use inline bracket citations instead."
    )


def _category_prefix(company_labels: List[str]) -> str:
    roster = "\n".join(f"- {label}" for label in company_labels)
    return (
        "CATEGORY QUERY SAFETY RULES\n"
        "Use only the exact company labels listed in the Retrieved Company Roster.\n"
        "Do not mention any company outside the roster anywhere in the answer, including Gaps / uncertainty or examples.\n"
        "Before writing the main comparison, check whether each roster company appears to match the user's requested category based ONLY on the retrieved context.\n"
        "If a roster company appears out of scope for the requested category, do NOT include it in the main comparison. Put it under 'Possible routing mismatch' with a short explanation and citation.\n"
        "Only describe how a company is addressing a risk if the retrieved text explicitly supports that claim. If the context describes the risk but not a concrete response, say that clearly.\n"
        "Use the exact company labels from the roster verbatim. Do not rename, shorten, substitute, or expand them.\n\n"
        "Retrieved Company Roster:\n"
        f"{roster}"
    )


def _multi_company_prefix(company_labels: List[str]) -> str:
    roster = "\n".join(f"- {label}" for label in company_labels)
    return (
        "STRICT COMPANY LABEL RULES\n"
        "Use only the exact company labels listed below when naming companies in the answer.\n"
        "Copy each label verbatim, including punctuation and the ticker in parentheses.\n"
        "Do not expand, shorten, substitute, or rename any company.\n"
        "Do not map a ticker to a different company.\n"
        "If a company is not in the roster below, do not mention it by name.\n"
        "If the question asks about a company but the retrieved context for that company is missing or weak, say: \"Insufficient evidence in retrieved context for <exact company label>.\"\n\n"
        "Company roster from retrieved context:\n"
        f"{roster}"
    )


def build_prompt(question: str, chunks: List[Dict], prompt_template_path: str) -> str:
    template = Path(prompt_template_path).read_text(encoding="utf-8")
    prompt = template.replace("{{question}}", question)
    prompt = prompt.replace("{{context}}", format_context(chunks))

    company_labels = _company_labels(chunks)

    if _is_single_company_temporal(question, company_labels):
        header = _single_company_temporal_prefix(company_labels[0])
    elif _is_category_query(question, company_labels):
        header = _category_prefix(company_labels)
    else:
        header = _multi_company_prefix(company_labels)

    return header + "\n\n" + prompt