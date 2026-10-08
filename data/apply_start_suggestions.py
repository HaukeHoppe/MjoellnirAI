"""
Puts the suggestions from start_suggestions.json (see generate_start_suggestions.py)
on the Capital Markets RAG model in Open WebUI, so they show on the start page of
a new chat with that model.

Stored as a model entry for the pipeline model id (what Workspace > Models >
Edit creates); an existing entry keeps its other settings. A new entry gets
public read access, like a model without an entry has.

With --description, the model's description is set too; Open WebUI shows it as Markdown
under the model name on the start page (e.g. a link to the project's repository).

Run inside the open-webui container:
    docker exec -w /app/backend open-webui python /data/apply_start_suggestions.py
Then reload the browser tab.
"""

import argparse
import asyncio
import json
import sys

sys.path.insert(0, "/app/backend")
from open_webui.models.models import ModelForm, Models  # noqa: E402
from open_webui.models.users import Users  # noqa: E402


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="/data/faiss_capital_index/start_suggestions.json")
    parser.add_argument("--model_id", default="capital_rag_pipeline")
    parser.add_argument("--name", default="Capital Markets RAG")
    parser.add_argument("--description", help="Markdown shown under the model name; omitted = left unchanged")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as f:
        suggestions = json.load(f)

    existing = await Models.get_model_by_id(args.model_id)
    if existing:
        meta = existing.meta.model_dump()
        meta["suggestion_prompts"] = suggestions
        if args.description is not None:
            meta["description"] = args.description
        form = ModelForm(
            id=existing.id,
            base_model_id=existing.base_model_id,
            name=existing.name,
            meta=meta,
            params=existing.params.model_dump(),
            is_active=existing.is_active,
        )
        result = await Models.update_model_by_id(args.model_id, form)
    else:
        admin = await Users.get_super_admin_user() or await Users.get_first_user()
        meta = {"suggestion_prompts": suggestions}
        if args.description is not None:
            meta["description"] = args.description
        form = ModelForm(
            id=args.model_id,
            name=args.name,
            meta=meta,
            params={},
            access_grants=[{"principal_type": "user", "principal_id": "*", "permission": "read"}],
        )
        result = await Models.insert_new_model(form, admin.id)

    if result is None:
        sys.exit(f"Failed to save suggestions on model {args.model_id}")
    print(f"{len(suggestions)} start suggestions set on {args.model_id} ({result.name})")


if __name__ == "__main__":
    asyncio.run(main())
