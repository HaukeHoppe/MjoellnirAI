# Open WebUI RAG Setup

Local Open WebUI stack with two Pipelines servers that serve custom RAG pipelines:

| Service             | Port | Purpose                                                   |
|---------------------|------|-----------------------------------------------------------|
| `open-webui`        | 3000 | Chat UI                                                   |
| `pipelines`         | 9099 | Climate PDF RAG (smoke-test pipeline)                     |
| `pipelines-capital` | 9098 | Capital Markets RAG (`pipelines-capital/capital_rag_pipeline.py`) |

See `architecture_diagram.png` for an overview and
[`docs/CAPITAL_RAG_PIPELINE.md`](docs/CAPITAL_RAG_PIPELINE.md) for a detailed description of the
Capital Markets RAG pipeline (ingestion, query workflow, hallucination checks, fallbacks, valves).
[`docs/CAPITAL_RAG_BEST_PRACTICES.md`](docs/CAPITAL_RAG_BEST_PRACTICES.md) explains the architecture and the
retrieval and grounding practices that make sure answers come only from the real source data.

## Setup

```sh
cp .env.example .env        # add your OPENAI_API_KEY
docker compose up -d --build
```

## Layout

- `pdfs/` – mounted at `/data`; ingestion scripts, the climate test pipeline and helpers.
  FAISS indexes are generated here and are not tracked in git.
- `pipelines-capital/` – Capital Markets RAG pipeline, loaded by the second Pipelines container.
- `all_rag_techniques_runnable_scripts/` – reference implementations of RAG techniques
  (fusion retrieval, HyPE, RAPTOR, reranking, ...) the pipelines are adapted from.

## Building the indexes

```sh
docker exec -it open-webui-pipelines python /data/ingest_climate_pdf.py
docker exec open-webui-pipelines-capital python /data/ingest_capital_chunks.py
```

Start-page suggestions for the capital model:

```sh
docker exec open-webui-pipelines-capital python /data/generate_start_suggestions.py
docker exec -w /app/backend open-webui sh -c 'WEBUI_SECRET_KEY="$(cat .webui_secret_key)" python /data/apply_start_suggestions.py'
```
