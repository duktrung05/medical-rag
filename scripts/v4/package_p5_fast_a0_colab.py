"""Build a scorer-compatible Colab bundle from mixed-lineage A0 pair shards."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path

from scripts.v4.prepare_p5_3a_colab_canary import DEPENDENCIES, SCORER_MODULES, archive_entry
from scripts.v4.run_full_baseline import _write_bytes_checked
from scripts.v4.run_p5_2_split import ROOT, digest, inputs, validate_pair_shard
from src.medrag.data.pair_lineage import PairLineageResolver, canonical_bytes
from src.medrag.data.phase2_batch import sha256_file

INPUT_DIR = Path("outputs/v4/a0_fast_parent_first_v1_colab_input")
MAX_INPUT_BYTES = 8_000_000_000


def package(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    _, _, queries, _, _ = inputs(ROOT / "configs/v4_submission_p2_a0.yaml", "full")
    query_by_id = {int(row["id"]): str(row["query"]) for row in queries}
    resolver = PairLineageResolver(source)
    merged = resolver.build_merged_manifest(query_by_id, require_complete=True)

    # Bundle shards get one derived transport lock so the existing external
    # scorer can validate them. Source and local on-disk shards remain unchanged.
    bundle_lock = copy.deepcopy(resolver.lock)
    bundle_lock["format"] = "medrag-v4-p5-2-split-lock-v1"
    bundle_lock["mode"] = "full"
    bundle_lock["transport_derivation"] = {
        "format": "medrag-v4-p5-fast-a0-colab-transport-v1",
        "optimized_lock_sha256": sha256_file(source / "lock.json"),
        "lineage_sha256": sha256_file(source / "lineage.json"),
        "candidate_pair_fields_unchanged": True,
        "only_rewritten_pair_field": "lock_sha256",
    }
    lock_bytes = canonical_bytes(bundle_lock)
    lock_sha = hashlib.sha256(lock_bytes).hexdigest()
    files: dict[str, bytes] = {str(INPUT_DIR / "lock.json"): lock_bytes}
    pair_hashes = {}
    provenance = {}
    pair_count = 0
    for qid in bundle_lock["query_ids"]:
        original, origin, original_sha = resolver.validate(qid, query_by_id[qid])
        transported = copy.deepcopy(original)
        transported["lock_sha256"] = lock_sha
        transport_bytes = canonical_bytes(transported)
        transport_sha = hashlib.sha256(transport_bytes).hexdigest()
        validate_pair_shard(transported, bundle_lock, qid, query_by_id[qid],
                            resolver.local_cache)
        files[str(INPUT_DIR / "pairs" / f"{qid}.json")] = transport_bytes
        pair_hashes[str(qid)] = transport_sha
        provenance[str(qid)] = {
            "origin": origin,
            "original_sha256": original_sha,
            "transport_sha256": transport_sha,
            "candidate_count": original["candidate_count"],
            "pair_input_hashes_sha256": digest("\n".join(
                item["input_sha256"] for item in original["pairs"]
            )),
        }
        pair_count += len(original["pairs"])
    if pair_count != 240_000:
        raise ValueError(f"Full A0 pair count differs: {pair_count}")
    pair_manifest = {
        "format": "medrag-v4-p5-2-pair-manifest-v1",
        "lock_sha256": lock_sha,
        "pair_shards_sha256": pair_hashes,
    }
    files[str(INPUT_DIR / "pair_manifest.json")] = canonical_bytes(pair_manifest)
    files[str(INPUT_DIR / "key_contract.sqlite")] = (source / "key_contract.sqlite").read_bytes()
    files[str(INPUT_DIR / "lineage_provenance.json")] = canonical_bytes({
        "format": "medrag-v4-p5-fast-a0-colab-lineage-v1",
        "merged_manifest_sha256": hashlib.sha256(canonical_bytes(merged)).hexdigest(),
        "shards": provenance,
    })
    code_files = [ROOT / name for name in SCORER_MODULES]
    code_files += [ROOT / "scripts/v4/run_full_baseline.py",
                   ROOT / "scripts/v4/run_p5_2_split.py",
                   ROOT / "scripts/v4/p5_3a_colab_orchestrator.py",
                   ROOT / "configs/v4_submission_p2_a0.yaml"]
    for path in code_files:
        files[str(path.relative_to(ROOT))] = path.read_bytes()
    files["p5_3a_requirements.txt"] = DEPENDENCIES.encode("ascii")
    total_bytes = sum(len(value) for value in files.values())
    if total_bytes > MAX_INPUT_BYTES:
        raise ValueError(f"Colab input bundle exceeds configured size: {total_bytes}")
    manifest = {
        "format": "medrag-v4-p5-fast-a0-colab-input-v1",
        "query_count": 1200,
        "pair_count": pair_count,
        "query_ids": bundle_lock["query_ids"],
        "input_dir": str(INPUT_DIR),
        "score_contract": bundle_lock["score_contract"],
        "transport_lock_sha256": lock_sha,
        "pair_shards_sha256": pair_hashes,
        "files_sha256": {name: hashlib.sha256(payload).hexdigest()
                         for name, payload in sorted(files.items())},
    }
    files["p5_3a_input_manifest.json"] = canonical_bytes(manifest)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", allowZip64=True) as bundle:
        for name, payload in sorted(files.items()):
            archive_entry(name, payload, bundle)
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "a0_fast_parent_first_v1_colab_bundle.zip"
    _write_bytes_checked(archive, stream.getvalue())
    _write_bytes_checked(output / "input_manifest.json", canonical_bytes(manifest))
    handoff = {
        "format": "medrag-v4-p5-fast-a0-colab-handoff-v1",
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "query_count": 1200,
        "pair_count": pair_count,
        "original_shard_bytes_modified": False,
        "transport_shards_derived_in_archive": True,
    }
    _write_bytes_checked(output / "handoff.json", canonical_bytes(handoff))
    return handoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
