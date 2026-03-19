import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import (
    DEFAULT_INDEX_DIR,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_PROMPT_PATH,
)
from src.llm import generate_with_ollama
from src.prompting import build_prompt, citation_label
from src.retrieval import HybridRetriever


class WarmRAGApp:
    """Loads retrieval artifacts once, then answers many questions in one process."""

    def __init__(
        self,
        index_dir: str,
        model: str,
        ollama_url: str,
        prompt_template: str,
        top_k: int = 8,
    ):
        start = time.perf_counter()
        self.retriever = HybridRetriever(index_dir)
        self.model = model
        self.ollama_url = ollama_url
        self.prompt_template = prompt_template
        self.top_k = top_k
        self.show_chunks = False
        self.show_prompt = False
        self.startup_seconds = time.perf_counter() - start

    def run_query(self, question: str):
        retrieve_start = time.perf_counter()
        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        prompt = build_prompt(question, chunks, self.prompt_template)
        retrieve_seconds = time.perf_counter() - retrieve_start

        if self.show_prompt:
            print("=" * 100)
            print("Final prompt")
            print("=" * 100)
            print(prompt)

        llm_start = time.perf_counter()
        answer = generate_with_ollama(prompt, model=self.model, url=self.ollama_url)
        llm_seconds = time.perf_counter() - llm_start
        total_seconds = retrieve_seconds + llm_seconds

        return {
            "chunks": chunks,
            "answer": answer,
            "retrieve_seconds": retrieve_seconds,
            "llm_seconds": llm_seconds,
            "total_seconds": total_seconds,
        }


def print_help() -> None:
    print(
        """
Commands:
  :quit / :exit      Exit
  :chunks on|off     Toggle retrieved chunk preview
  :prompt on|off     Toggle final prompt preview
  :topk N            Change top-k for retrieval
  :help              Show this help

Any other input is treated as a question.
""".strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent warm-process CLI for SEC RAG demos.")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--prompt-template", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--chunks", action="store_true", help="Show retrieved chunks for each query")
    parser.add_argument("--prompt", action="store_true", help="Show the final prompt for each query")
    args = parser.parse_args()

    app = WarmRAGApp(
        index_dir=args.index_dir,
        model=args.model,
        ollama_url=args.ollama_url,
        prompt_template=args.prompt_template,
        top_k=args.top_k,
    )
    app.show_chunks = args.chunks
    app.show_prompt = args.prompt

    print("=" * 100)
    print("Warm SEC RAG CLI")
    print("=" * 100)
    print(f"startup time: {app.startup_seconds:.2f}s")
    print(f"index dir    : {args.index_dir}")
    print(f"model        : {args.model}")
    print(f"top_k        : {app.top_k}")
    print("Type :help for commands. Ask questions below.\n")

    while True:
        try:
            raw = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue
        if raw in {":quit", ":exit"}:
            print("Exiting.")
            break
        if raw == ":help":
            print_help()
            continue
        if raw.startswith(":chunks "):
            value = raw.split(maxsplit=1)[1].lower()
            app.show_chunks = value == "on"
            print(f"retrieved chunk preview: {'on' if app.show_chunks else 'off'}")
            continue
        if raw.startswith(":prompt "):
            value = raw.split(maxsplit=1)[1].lower()
            app.show_prompt = value == "on"
            print(f"prompt preview: {'on' if app.show_prompt else 'off'}")
            continue
        if raw.startswith(":topk "):
            try:
                app.top_k = int(raw.split(maxsplit=1)[1])
                print(f"top_k set to {app.top_k}")
            except ValueError:
                print("Please provide an integer, e.g. :topk 6")
            continue

        try:
            result = app.run_query(raw)
            print("\n")
            print("=" * 100)
            print("Timing")
            print("=" * 100)
            print(f"retrieval + prompt build: {result['retrieve_seconds']:.2f}s")
            print(f"llm generation        : {result['llm_seconds']:.2f}s")
            print(f"total query time      : {result['total_seconds']:.2f}s\n")

            if app.show_chunks:
                print("=" * 100)
                print("Retrieved chunks")
                print("=" * 100)
                for i, chunk in enumerate(result["chunks"], start=1):
                    print(f"{i}. {citation_label(chunk)}")
                    print(chunk["text"][:280].replace("\n", " ") + "...")
                    print()

            print("=" * 100)
            print("Answer")
            print("=" * 100)
            print(result["answer"])
            print()
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()