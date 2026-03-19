from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "index"
DEFAULT_PROMPT_PATH = REPO_ROOT / "prompts" / "final_prompt.txt"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = "gemma3:4b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

CHUNK_SIZE_CHARS = 1800
CHUNK_OVERLAP_CHARS = 250
DENSE_TOP_K = 12
BM25_TOP_K = 12
FINAL_TOP_K = 8
COMPANY_ROUTE_TOP_K = 5
RRF_K = 60
MAX_CHUNKS_PER_COMPANY = 3
MAX_CHUNKS_PER_SECTION = 2

ITEM_TITLES_10K = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Common Equity",
    "6": "Reserved",
    "7": "MD&A",
    "7A": "Market Risk",
    "8": "Financial Statements",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Foreign Jurisdictions Preventing Inspections",
    "10": "Directors and Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

ITEM_TITLES_10Q = {
    "1": "Financial Statements / Legal Proceedings",
    "1A": "Risk Factors",
    "2": "MD&A / Unregistered Sales",
    "3": "Market Risk / Defaults",
    "4": "Controls and Procedures / Mine Safety",
    "5": "Other Information",
    "6": "Exhibits",
}


def item_title(form_type: str, item_code: str) -> str:
    form_type = form_type.upper().replace("-", "")
    if form_type == "10K":
        return ITEM_TITLES_10K.get(item_code, "Unknown Section")
    return ITEM_TITLES_10Q.get(item_code, "Unknown Section")