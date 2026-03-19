import re
from typing import Dict, List, Tuple

from .config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS, item_title

ITEM_RE_10K = re.compile(r"\bItem\s+(1A|1B|1C|1|2|3|4|5|6|7A|7|8|9A|9B|9C|9|10|11|12|13|14|15|16)\.?", re.I)
ITEM_RE_10Q = re.compile(r"\bItem\s+(1A|1|2|3|4|5|6)\.?", re.I)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\xa0", " ")
    # Many filings flatten section headers like `PART IItem 1.` or `17Item 2.`.
    text = re.sub(r"([A-Za-z0-9\)])(Item\s+[0-9])", r"\1 \2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_body_start(text: str) -> int:
    """
    SEC txt exports often begin with metadata and a table of contents.
    Prefer the first real `Part I ... Item 1` body heading. If that fails,
    fall back to the first non-TOC `Item 1` occurrence.
    """
    for m in re.finditer(r"PART\s+I(?!,)[^|]{0,120}?Item\s+1\.?", text, re.I | re.S):
        preview = text[m.start(): m.start() + 180]
        if "|" not in preview[:120]:
            return max(0, m.start())

    for m in re.finditer(r"\bItem\s+1\.?", text, re.I):
        preview = text[m.start(): m.start() + 140]
        if "|" not in preview[:100]:
            return max(0, m.start() - 20)
    return 0


def detect_item_positions(text: str, form_type: str) -> List[Tuple[int, str]]:
    pattern = ITEM_RE_10K if form_type.upper() == "10K" else ITEM_RE_10Q
    out: List[Tuple[int, str]] = []
    last_pos = -10_000
    for m in pattern.finditer(text):
        preview = text[m.start(): m.start() + 100]
        if "|" in preview[:60]:
            continue
        if m.start() - last_pos < 200:
            continue
        item_code = m.group(1).upper()
        out.append((m.start(), item_code))
        last_pos = m.start()
    return out


def make_chunks(meta: Dict[str, str], raw_text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[Dict]:
    body_start = find_body_start(raw_text)
    body_text = normalize_whitespace(raw_text[body_start:])
    item_positions = detect_item_positions(body_text, meta["form_type"])

    chunks: List[Dict] = []
    step = max(1, chunk_size - overlap)
    chunk_idx = 0

    for start in range(0, len(body_text), step):
        end = min(len(body_text), start + chunk_size)
        chunk_text = body_text[start:end].strip()
        if len(chunk_text) < 250:
            continue

        latest_item = ""
        for pos, code in item_positions:
            if pos <= start:
                latest_item = code
            else:
                break
        if not latest_item and item_positions and item_positions[0][0] < 120:
            latest_item = item_positions[0][1]

        section_title = item_title(meta["form_type"], latest_item) if latest_item else "Unknown Section"

        chunks.append(
            {
                "chunk_id": f"{meta['ticker']}_{meta['form_type']}_{meta.get('filing_date', 'unknown')}_{chunk_idx}",
                "ticker": meta.get("ticker", ""),
                "company": meta.get("company", meta.get("ticker", "")),
                "form_type": meta.get("form_type", ""),
                "filing_date": meta.get("filing_date", ""),
                "report_period": meta.get("report_period", ""),
                "quarter": meta.get("quarter", ""),
                "cik": meta.get("cik", ""),
                "url": meta.get("url", ""),
                "section_code": latest_item,
                "section_title": section_title,
                "start_char": start,
                "end_char": end,
                "text": chunk_text,
            }
        )
        chunk_idx += 1

        if end >= len(body_text):
            break

    return chunks
