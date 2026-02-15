import asyncio
import logging
from indexer import get_current_job

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    async def main():
        print("Starting standalone indexer...")
        job = get_current_job()
        status = await job.start()
        print(f"Indexing finished with status: {status}")
        if job.errors:
            print(f"Errors encountered: {len(job.errors)}")
            for err in job.errors:
                print(f" - {err}")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInput cancelled.")
