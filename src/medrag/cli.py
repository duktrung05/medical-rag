from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.medrag.core.config import load_config
from src.medrag.core.paths import project_root
from src.medrag.data.corpus_registry import CorpusRegistry
from src.medrag.data.prepared import load_prepared, prepare_from_official


def main() -> None:
    parser = argparse.ArgumentParser(prog="medrag-v4")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pilot")
    sub.add_parser("check-config")
    prepare = sub.add_parser("prepare", help="Prepare a bounded, pinned official corpus snapshot")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--limit", type=int, default=100)
    prepare_phase2 = sub.add_parser("prepare-phase2", help="Prepare a locked Phase 2 release subset for V4")
    prepare_phase2.add_argument("--config", type=Path,
                                default=Path("configs/vibiomir/phase2_batch_migration_v1.json"))
    prepare_phase2.add_argument("--doc-ids", type=Path, required=True,
                                help="JSON array of explicit eligible Phase 1/2 document IDs")
    prepare_phase2.add_argument("--query-ids", type=Path, required=True,
                                help="JSON array of explicit official query IDs")
    prepare_phase2.add_argument("--output", type=Path, required=True,
                                help="New snapshot directory under outputs/v4")
    search = sub.add_parser("search", help="Search a prepared V4 snapshot")
    search.add_argument("--prepared", type=Path, required=True,
                        help="V4 prepared JSONL file or V4 Phase 2 snapshot directory")
    search.add_argument("--config", type=Path, help="V4 config; defaults to configs/v4_core.yaml")
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    root = project_root()
    config_path = (args.config if args.command == "search" and args.config else
                   root / "configs/v4_core.yaml")
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)
    registry = CorpusRegistry.load(root / config["registry"])
    version = registry.get(config["corpus_version"])
    if args.command == "check-config":
        verified = CorpusRegistry.verify_file(version)
        print(json.dumps({"version": version.name, "revision": version.revision,
                          "path_exists": version.path.exists(), "sha256_match": verified}))
        if not verified:
            raise SystemExit("Corpus snapshot SHA-256 mismatch")
    elif args.command == "pilot":
        from src.medrag.pilot import main as pilot
        pilot()
    elif args.command == "prepare":
        reuse = registry.get("legacy-local-20261006")
        result = prepare_from_official(version, reuse, root / "data/vibiomir/crawl/pages",
                                       args.output, args.limit)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "prepare-phase2":
        from src.medrag.data.phase2_batch import load_phase2_batch
        from src.medrag.data.phase2_prepared import write_phase2_prepared

        phase2_config = args.config if args.config.is_absolute() else root / args.config
        doc_ids_path = args.doc_ids if args.doc_ids.is_absolute() else root / args.doc_ids
        query_ids_path = args.query_ids if args.query_ids.is_absolute() else root / args.query_ids
        output_path = args.output if args.output.is_absolute() else root / args.output
        doc_ids = json.loads(doc_ids_path.read_text(encoding="utf-8"))
        query_ids = json.loads(query_ids_path.read_text(encoding="utf-8"))
        batch = load_phase2_batch(phase2_config, doc_ids=doc_ids, query_ids=query_ids,
                                  max_documents=1000)
        result = write_phase2_prepared(batch, output_path, version)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "search":
        if args.limit < 1:
            parser.error("--limit must be positive")
        prepared_path = args.prepared if args.prepared.is_absolute() else root / args.prepared
        pipeline = load_prepared(prepared_path, registry, config=config)
        print(json.dumps(pipeline.retrieve(args.query, args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
