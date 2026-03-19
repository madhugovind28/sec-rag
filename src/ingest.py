import glob
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

FILENAME_RE = re.compile(
    r"^(?P<ticker>[A-Z]+)_(?P<form>10K|10Q)_(?:(?P<quarter>\d{4}Q\d)_)?(?P<filing_date>\d{4}-\d{2}-\d{2})_full\.txt$"
)

HEADER_FIELDS = {
    "Company:": "company",
    "Ticker:": "ticker",
    "Filing Type:": "filing_type_raw",
    "Filing Date:": "filing_date",
    "Report Period:": "report_period",
    "Quarter:": "quarter",
    "CIK:": "cik",
    "URL:": "url",
}


def prepare_corpus(corpus_path: str, extract_dir: str) -> Path:
    corpus = Path(corpus_path)
    extract_root = Path(extract_dir)
    extract_root.mkdir(parents=True, exist_ok=True)

    if corpus.is_dir():
        return corpus

    if corpus.suffix.lower() != ".zip":
        raise ValueError(f"Expected a directory or .zip corpus, got: {corpus}")

    target = extract_root / corpus.stem
    if target.exists() and any(target.iterdir()):
        return target

    with zipfile.ZipFile(corpus, "r") as zf:
        zf.extractall(target)
    return target


def iter_filing_paths(corpus_dir: str) -> List[Path]:
    manifest = Path(corpus_dir) / "manifest.json"
    if manifest.exists():
        import json
        files = json.loads(manifest.read_text(encoding="utf-8")).get("files", [])
        return [Path(corpus_dir) / name for name in files if str(name).endswith(".txt")]
    return [Path(p) for p in sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))]


def parse_filename(path: Path) -> Dict[str, str]:
    m = FILENAME_RE.match(path.name)
    if not m:
        return {
            "ticker": path.name.split("_")[0],
            "form_type": "10K" if "10K" in path.name else "10Q",
            "quarter": "",
            "filing_date": "",
        }
    out = m.groupdict()
    return {
        "ticker": out["ticker"],
        "form_type": out["form"],
        "quarter": out.get("quarter") or "",
        "filing_date": out["filing_date"],
    }


def read_filing(path: Path) -> Tuple[Dict[str, str], str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    meta = parse_filename(path)

    for line in text.splitlines()[:20]:
        for prefix, key in HEADER_FIELDS.items():
            if line.startswith(prefix):
                meta[key] = line.split(":", 1)[1].strip()

    filing_type_raw = meta.get("filing_type_raw", "")
    if "10-Q" in filing_type_raw.upper():
        meta["form_type"] = "10Q"
    elif "10-K" in filing_type_raw.upper():
        meta["form_type"] = "10K"

    meta["source_path"] = str(path)
    return meta, text
