import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.chunking import make_chunks
from src.config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS, DEFAULT_EMBED_MODEL
from src.indexing import build_indices
from src.ingest import iter_filing_paths, prepare_corpus, read_filing
from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a minimal hybrid RAG index over SEC filings.")
    parser.add_argument("--corpus", required=True, help="Path to edgar_corpus.zip or extracted corpus directory")
    parser.add_argument("--out", default=str(ROOT / "data" / "index"), help="Output index directory")
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / ".cache"), help="Where to extract the zip")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="SentenceTransformers embedding model")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_CHARS)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP_CHARS)
    args = parser.parse_args()

    corpus_dir = prepare_corpus(args.corpus, args.cache_dir)
    filing_paths = iter_filing_paths(str(corpus_dir))
    if not filing_paths:
        raise SystemExit(f"No .txt filings found in {corpus_dir}")

    all_chunks = []
    for path in filing_paths:
        meta, raw_text = read_filing(path)
        chunks = make_chunks(meta, raw_text, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
        all_chunks.extend(chunks)

    ensure_dir(Path(args.out))
    build_indices(all_chunks, args.out, args.embed_model)
    print(f"Built index at {args.out}")
    print(f"Indexed {len(filing_paths)} filings into {len(all_chunks)} chunks")


if __name__ == "__main__":
    main()
