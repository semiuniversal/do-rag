#!/usr/bin/env python3
# index_docs.py
# CLI wrapper for the indexing engine.
# For MCP-based indexing, use the start_indexing/stop_indexing/get_indexing_status tools instead.

import argparse
import asyncio
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("indexing.log", mode="w"),
        logging.StreamHandler(),
    ],
)

from indexer import IndexingJob


async def run_with_progress(reset: bool):
    """Run indexing with periodic progress output."""
    job = IndexingJob()
    task = asyncio.create_task(job.start(reset=reset))

    # Poll for progress while the job runs
    while not task.done():
        await asyncio.sleep(2)
        if job.status in ("scanning", "indexing"):
            print(f"\r{job.get_status_text()}", end="", flush=True, file=sys.stderr)

    print("", file=sys.stderr)  # newline after progress
    result = await task
    print(result, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Index local documents for RAG")
    parser.add_argument("--reset", action="store_true", help="Reset index before indexing")
    parser.add_argument("--force", action="store_true", help="Skip confirmation for --reset")
    args = parser.parse_args()

    if args.reset and not args.force:
        import config
        print(f"\n⚠️  WARNING: You are about to ERASE the entire index at: {config.CHROMA_PERSIST_DIRECTORY}")
        print("This action cannot be undone.")
        confirm = input("Are you sure you want to continue? [y/N]: ")
        if confirm.lower() != "y":
            print("Reset cancelled.")
            sys.exit(0)

    start = time.time()
    asyncio.run(run_with_progress(reset=args.reset))
    elapsed = time.time() - start
    logging.info(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
