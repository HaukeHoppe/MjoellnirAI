"""
Offline ingestion smoke-test: builds a FAISS index from Understanding_Climate_Change.pdf.

This is NOT the real capital-markets ingestion pipeline (see
RAG_Techniques/PROMPT_openwebui_pipeline.md for that). It exists to validate the
Docker/Open WebUI Pipelines plumbing end-to-end with a simple static PDF before
building semantic_chunking -> concept/opinion classification -> ... over real
YouTube transcripts.

Run inside the pipelines container, where requirements.txt is already installed
and OPENAI_API_KEY is already set as an env var:

    docker exec -it open-webui-pipelines python /data/ingest_climate_pdf.py
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper_functions import encode_pdf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PDF = os.path.join(THIS_DIR, "Understanding_Climate_Change.pdf")
DEFAULT_INDEX_DIR = os.path.join(THIS_DIR, "faiss_climate_index")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from the climate change PDF."
    )
    parser.add_argument("--path", default=DEFAULT_PDF)
    parser.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set in the environment. "
            "It should already be set via docker-compose.yaml for this container."
        )

    print(f"Encoding {args.path} ...")
    vectorstore = encode_pdf(
        args.path, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    vectorstore.save_local(args.index_dir)
    print(f"Saved FAISS index to {args.index_dir}")


if __name__ == "__main__":
    main()
