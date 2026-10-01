"""Freeze the current ViMed validation rankings and their provenance."""

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

from scripts.workflows.vimedqa.benchmark_vimed_validation import CONFIGS, read_rows
from src.data.loader import DataLoader
from src.evaluation.ranking_metrics import summarize_rankings
from src.indexing.sparse_index import corpus_sha256


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = f"baseline_{date.today():%Y%m%d}"
CONFIG_ARCHIVE = ROOT / "configs" / "archive" / ARCHIVE_NAME
SNAPSHOT = ROOT / "outputs" / "baseline_reference" / ARCHIVE_NAME


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if CONFIG_ARCHIVE.exists() or SNAPSHOT.exists():
        raise FileExistsError(f"Baseline destination already exists: {ARCHIVE_NAME}")

    samples_path = ROOT / "data/vimed/validation/samples.jsonl"
    corpus_path = ROOT / "data/vimed/chunks.jsonl"
    samples = read_rows(samples_path)
    chunks = DataLoader.load_chunks(corpus_path)
    corpus_hash = corpus_sha256(chunks)
    source_manifest = json.loads((ROOT / "data/vimed/validation/manifest.json").read_text())
    samples_hash = hashlib.sha256(json.dumps(samples, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if source_manifest["split"] != "validation" or source_manifest["corpus_sha256"] != corpus_hash:
        raise ValueError("Validation/corpus identity mismatch")

    CONFIG_ARCHIVE.mkdir(parents=True)
    SNAPSHOT.mkdir(parents=True)
    config_records = {}
    metric_records = {}
    for stage, relative_config in CONFIGS.items():
        config_path = ROOT / relative_config
        archived_config = CONFIG_ARCHIVE / Path(relative_config).name
        shutil.copy2(config_path, archived_config)
        config_records[stage] = {
            "path": str(archived_config.relative_to(ROOT)),
            "sha256": sha256(config_path),
        }

        source_dir = ROOT / "outputs/vimed_validation" / stage
        target_dir = SNAPSHOT / stage
        target_dir.mkdir()
        for filename in ("rankings.jsonl", "metrics.json", "manifest.json", "run_identity.json"):
            source = source_dir / filename
            shutil.copy2(source, target_dir / filename)

        rows = read_rows(source_dir / "rankings.jsonl")
        saved_metrics = json.loads((source_dir / "metrics.json").read_text())
        replay_metrics = summarize_rankings(samples, rows)
        mismatch = {key: {"saved": saved_metrics.get(key), "replayed": value}
                    for key, value in replay_metrics.items() if saved_metrics.get(key) != value}
        if mismatch:
            raise ValueError(f"{stage} metric replay mismatch: {mismatch}")
        manifest = json.loads((source_dir / "manifest.json").read_text())
        identity = json.loads((source_dir / "run_identity.json").read_text())
        config_values = __import__("yaml").safe_load(config_path.read_text())
        retrieval_values = config_values.get("retrieval", {})
        metric_records[stage] = {
            "num_queries": saved_metrics["num_queries"],
            **{key: saved_metrics[key] for key in (
                "recall_at_k", "document_recall_at_k", "mrr_at_100", "zero_hit_queries_at_k", "by_topic",
                "latency_ms", "candidate_depth", "latency_method",
                "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes") if key in saved_metrics},
            "replay_verified": True,
            "rankings_sha256": sha256(source_dir / "rankings.jsonl"),
            "config_sha256_current_and_manifest": sha256(config_path),
            "run_identity_config_sha256": identity.get("config_sha256"),
            "run_identity_hash_matches_current": identity.get("config_sha256") == sha256(config_path),
            "manifest_config_hash_matches_current": manifest.get("config_sha256") == sha256(config_path),
            "retriever_top_k": {name: value.get("top_k") for name, value in retrieval_values.items()},
            "fusion_top_k": config_values.get("fusion", {}).get("top_k"),
            "reranker_top_k": config_values.get("reranker", {}).get("top_k") if config_values.get("reranker", {}).get("enabled") else None,
        }

    selected_path = ROOT / "configs/vimed_selected_validation.yaml"
    selected_archive = CONFIG_ARCHIVE / selected_path.name
    shutil.copy2(selected_path, selected_archive)
    config_records["selected"] = {
        "path": str(selected_archive.relative_to(ROOT)),
        "sha256": sha256(selected_path),
    }

    # The reranker and dense model use immutable model commits. Tokenizer revisions
    # were not recorded in the source artifacts and are deliberately left unknown.
    rerank_config = __import__("yaml").safe_load((CONFIG_ARCHIVE / "vimed_hybrid_rerank.yaml").read_text())
    dense_config = __import__("yaml").safe_load((CONFIG_ARCHIVE / "vimed_dense_bge_m3.yaml").read_text())
    provenance = {
        "baseline_id": ARCHIVE_NAME,
        "source_commit_sha": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dataset": "ViMedQA closed corpus",
        "split": "validation",
        "num_queries": len(samples),
        "num_chunks": len(chunks),
        "corpus_sha256": corpus_hash,
        "samples_sha256": samples_hash,
        "validation_manifest": source_manifest,
        "configs": config_records,
        "models": {
            "dense": {"name": dense_config["retrieval"]["dense"]["model_name"],
                      "revision": dense_config["retrieval"]["dense"]["revision"],
                      "tokenizer_revision": "unknown (not recorded in source artifact)"},
            "reranker": {"name": rerank_config["reranker"]["model_name"],
                         "revision": rerank_config["reranker"]["revision"],
                         "tokenizer_revision": "unknown (not recorded in source artifact)"},
        },
        "stages": metric_records,
        "provenance_notes": [
            "Artifacts are historical validation rankings; replay verified saved ranking metrics.",
            "Some hybrid/rerank run_identity config hashes differ from the current config and manifest hashes; all values are preserved.",
            "Latency for hybrid and rerank is cached-stage sum plus current stage, not direct end-to-end API latency.",
            "Tokenizer revisions are not present in the source artifacts.",
        ],
    }
    (SNAPSHOT / "baseline_metrics.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    files = {}
    for path in sorted(SNAPSHOT.rglob("*")):
        if path.is_file() and path.name != "snapshot_manifest.json":
            files[str(path.relative_to(ROOT))] = sha256(path)
    snapshot_manifest = {"baseline_id": ARCHIVE_NAME, "source_commit_sha": provenance["source_commit_sha"],
                         "files_sha256": files}
    (SNAPSHOT / "snapshot_manifest.json").write_text(json.dumps(snapshot_manifest, indent=2) + "\n")

    lines = [
        f"# Baseline tham chiếu ViMed — {ARCHIVE_NAME}", "",
        f"Commit nguồn: `{provenance['source_commit_sha']}`. Split: validation, {len(samples)} queries, {len(chunks)} chunks.",
        f"Corpus SHA256: `{corpus_hash}`.", "",
        "Baseline retrieval được đóng băng từ rankings lịch sử của bốn stage. Metrics được replay từ rankings bằng CPU và khớp artifact đã lưu.", "",
        "| Stage | Recall@1 | @3 | @5 | @10 | @20 | @50 | @100 | MRR@100 | p95 ms | VRAM peak allocated | Zero hit @10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, values in metric_records.items():
        recall = values["recall_at_k"]
        latency = values["latency_ms"]
        lines.append(f"| {stage} | " + " | ".join(
            [f"{recall[str(k)]:.4f}" for k in (1, 3, 5, 10, 20, 50, 100)] +
            [f"{values['mrr_at_100']:.4f}", f"{latency['p95']:.1f}",
             str(values.get("peak_cuda_allocated_bytes", "unknown")),
             str(values["zero_hit_queries_at_k"]["10"])]) + " |")
    lines += ["", "Document Recall@K và số zero-hit ở mọi cutoff nằm trong `outputs/baseline_reference/" + ARCHIVE_NAME + "/baseline_metrics.json`.",
              "", "Năm cấu hình gốc được sao lưu nguyên văn trong `configs/archive/" + ARCHIVE_NAME + "/`. Rankings, metrics, manifest và run identity được sao lưu riêng theo stage.",
              "", "Giới hạn provenance: tokenizer revision chưa được ghi; hash `run_identity` của hybrid/rerank khác hash config hiện tại và hash trong manifest. Các giá trị lịch sử được lưu nguyên trạng. Hybrid/rerank latency là tổng stage đã cache, không phải latency API trực tiếp.",
              "", "Replay CPU kiểm chứng metrics ranking. Tái chạy GPU chưa được thực hiện; test baseline hiện bị chặn khi import vì môi trường Python 3.14 thiếu `torch`.", ""]
    (ROOT / "docs" / "baseline_reference_vi.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"snapshot": str(SNAPSHOT.relative_to(ROOT)), "config_archive": str(CONFIG_ARCHIVE.relative_to(ROOT)),
                      "stages": {stage: {"recall_at_10": values["recall_at_k"]["10"],
                                         "replay_verified": values["replay_verified"],
                                         "identity_hash_matches": values["run_identity_hash_matches_current"]}
                                 for stage, values in metric_records.items()}}, indent=2))


if __name__ == "__main__":
    main()
