"""Command line interface: `hybrid-rag ingest | ask | chat | graph`."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .pipeline import Answer, HybridRAG

_COLORS = {"vector": "\033[96m", "graph": "\033[95m", "hybrid": "\033[93m"}
_DIM, _BOLD, _RESET = "\033[2m", "\033[1m", "\033[0m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if sys.stdout.isatty() else text


def print_answer(ans: Answer, show_context: bool = False) -> None:
    color = _COLORS.get(ans.route, "")
    print(_c(f"\n● Route: {ans.route.upper()}", color + _BOLD), _c(f"— {ans.route_reason}", _DIM))
    if ans.entities:
        print(_c(f"● Graph entities: {', '.join(ans.entities)}", _DIM))
    print(_c(f"● Roles: {', '.join(ans.roles)}", _DIM))
    print(f"\n{ans.answer}\n")
    print(_c("Sources:", _BOLD))
    for c in ans.contexts:
        extra = f"  (similarity {c.raw_score:.2f})" if c.kind == "doc" else ""
        print(f"  [{c.ref}] {c.citation}{_c(extra, _DIM)}")
        if show_context:
            print(_c("       " + " ".join(c.text.split())[:300], _DIM))
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hybrid-rag", description="Enterprise Hybrid RAG (Vector + Graph) with GPT")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Chunk + embed documents and build the knowledge graph")
    p_ingest.add_argument("--extract-graph", action="store_true", help="Also extract triples from documents with GPT")

    def add_query_args(p):
        p.add_argument("--role", action="append", help="User role (repeatable). Default: staff. 'admin' sees all.")
        p.add_argument("--route", choices=["auto", "vector", "graph", "hybrid"], default="auto")
        p.add_argument("--show-context", action="store_true")

    p_ask = sub.add_parser("ask", help="Ask one question")
    p_ask.add_argument("question")
    p_ask.add_argument("--json", action="store_true", help="Print the full result as JSON")
    add_query_args(p_ask)

    add_query_args(sub.add_parser("chat", help="Interactive question loop"))
    sub.add_parser("graph", help="Show knowledge graph statistics")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    try:
        rag = HybridRAG()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.cmd == "ingest":
        print(json.dumps(rag.ingest(extract_graph=args.extract_graph), indent=2))
        return 0

    rag.load()
    if args.cmd == "graph":
        print(json.dumps(rag.graph.stats(), indent=2))
        return 0

    route = None if args.route == "auto" else args.route
    if args.cmd == "ask":
        ans = rag.query(args.question, roles=args.role, route=route)
        print(json.dumps(ans.to_dict(), indent=2)) if args.json else print_answer(ans, args.show_context)
        return 0

    print("Ask about DailyBasket policies, suppliers, stores and managers. Type 'exit' to quit.")
    while True:
        try:
            question = input(_c("\nYou › ", _BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit", "q"}:
            return 0
        if question:
            print_answer(rag.query(question, roles=args.role, route=route), args.show_context)


if __name__ == "__main__":
    raise SystemExit(main())
