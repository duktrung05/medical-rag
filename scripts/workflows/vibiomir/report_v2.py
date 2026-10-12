"""Generate evidence-bounded Vietnamese V2 pilot reports from actual artifacts."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import shutil
import sqlite3
import statistics
import subprocess
from pathlib import Path

from r2ai.validate import validate as validate_official_schema
from src.retrieval_v2.core import passage_id

ROOT = Path(__file__).resolve().parents[3]


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def checksum(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_passages(db: Path) -> tuple[int, int]:
    if not db.exists():
        return 0, 0
    con = sqlite3.connect(db)
    valid = total = 0
    for doc_id, chunk_index, start, end, body, pid, source in con.execute(
        """SELECT p.doc_id,p.chunk_index,p.start_char,p.end_char,p.text,p.passage_id,s.text
        FROM passages p JOIN source_chunks s
        ON p.doc_id=s.doc_id AND p.chunk_index=s.chunk_index"""
    ):
        total += 1
        valid += body == source[start:end] and pid == passage_id(doc_id, chunk_index, start, end, body)
    con.close()
    return valid, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "outputs/retrieval_v2")
    ap.add_argument("--pilot", type=Path)
    ap.add_argument("--extended", type=Path)
    args = ap.parse_args()
    root = args.root
    pilot = args.pilot or root / "pilot"
    extended = args.extended or root / "pilot_full_docs"
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    corpus = read_json(pilot / "V2_CORPUS_MANIFEST.json")
    extended_corpus = read_json(extended / "V2_CORPUS_MANIFEST.json")
    audit = read_json(extended / "V2_DATA_AUDIT.json")
    index = read_json(pilot / "V2_INDEX_MANIFEST.json")
    extended_index = read_json(extended / "V2_INDEX_MANIFEST.json")
    env = read_json(pilot / "environment.json")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    benchmark = read_json(pilot / "V2_RETRIEVAL_BENCHMARK.json")
    predictions = pilot / "V2_FINAL_PREDICTIONS.jsonl"
    reranked = pilot / "V2_RERANKED_CANDIDATES.parquet"
    preview = pilot / "V2_SCHEMA_PREVIEW_NOT_FOR_SUBMISSION.zip"
    schema_errors = ["preview_missing"]
    schema_output = ""
    if preview.exists():
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            schema_errors = validate_official_schema(preview)
        schema_output = buffer.getvalue()
    write(reports / "V2_SUBMISSION_VALIDATION.md", f"""
# Kiểm tra định dạng submission

`r2ai.validate.validate` trả {len(schema_errors)} lỗi. ZIP preview chứa 1.200 query, 1.150 query ngoài pilot cố ý rỗng; không dùng để nộp hoặc đánh giá leaderboard.

```text
{schema_output.strip()}
```
""")
    complete = (predictions.exists() and reranked.exists() and
                (pilot / "v2_dense.faiss").exists() and not schema_errors)
    status = "V2_PILOT_RUNNING" if complete else "V2_PARTIALLY_IMPLEMENTED"
    n_queries = len(benchmark) if isinstance(benchmark, list) else 0
    run_manifest = {"status": status, "pilot_source_sha256": checksum(pilot / "pilot_source.parquet"),
                    "sparse_index_sha256": checksum(pilot / "v2_sparse.sqlite"),
                    "dense_index_sha256": checksum(pilot / "v2_dense.faiss"),
                    "predictions_sha256": checksum(predictions),
                    "config_sha256": checksum(ROOT / "configs/vibiomir/retrieval_v2.yaml"),
                    "query_count": n_queries, "schema_validation_errors": len(schema_errors),
                    "official_scoring": "PENDING_OFFICIAL_SCORING"}
    (root / "V2_RUN_MANIFEST.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    avg_latency = (statistics.mean(x["latency_seconds"] for x in benchmark) if n_queries else None)
    global_docs = (statistics.mean(x["global_unique_docs"] for x in benchmark) if n_queries else None)
    diverse_docs = (statistics.mean(x["diverse_unique_docs"] for x in benchmark) if n_queries else None)
    truncated = total_reranked = 0
    if reranked.exists():
        import pyarrow.parquet as pq
        table = pq.read_table(reranked, columns=["truncated"])
        flags = table.column("truncated").to_pylist()
        total_reranked, truncated = len(flags), sum(flags)
    doc_scores_path = pilot / "V2_DOCUMENT_SCORING.csv"
    doc_score_rows = list(csv.DictReader(doc_scores_path.open(encoding="utf-8"))) if doc_scores_path.exists() else []
    multi_support = sum(int(x["supporting_passages"]) > 1 for x in doc_score_rows)
    dense_bytes = (pilot / "v2_dense.faiss").stat().st_size if (pilot / "v2_dense.faiss").exists() else 0
    sparse_bytes = (pilot / "v2_sparse.sqlite").stat().st_size if (pilot / "v2_sparse.sqlite").exists() else 0
    full_sparse_bytes = (extended / "v2_sparse.sqlite").stat().st_size if (extended / "v2_sparse.sqlite").exists() else 0
    grounded_smoke = verify_passages(pilot / "v2_sparse.sqlite")
    grounded_extended = verify_passages(extended / "v2_sparse.sqlite")
    lexical_ablation = extended / "V2_V1_LEXICAL_PILOT_ABLATION.csv"
    lexical_rows = list(csv.DictReader(lexical_ablation.open(encoding="utf-8"))) if lexical_ablation.exists() else []
    v2_only = (statistics.mean(int(x["v2_only_docs"]) for x in lexical_rows) if lexical_rows else None)
    queries_v2_only = sum(int(x["v2_only_docs"]) > 0 for x in lexical_rows)

    write(reports / "V2_ARCHITECTURE.md", f"""
# Retrieval V2: kiến trúc pilot

Nguồn là extracted Parquet tại revision `{corpus.get('source_revision')}`. Mỗi passage là substring nguyên văn của đúng một `(doc_id, chunk_index)`, có offset ký tự trong chunk và SHA-256 identity. Không suy diễn offset của HTML gốc.

Hai index FTS5 độc lập: `document_fts` chứa title + toàn bộ từng source chunk rồi gộp điểm về document; `passage_fts` chứa title + cửa sổ passage. Analyzer tạo token Unicode và bigram Hán. Dense BGE-M3 mã hóa passage, chuẩn hóa L2 và tìm bằng FAISS inner product. RRF hợp nhất theo rank với provenance từng nhánh; phân bổ một passage cho mỗi document trước khi dùng budget còn lại. BGE reranker chấm tối đa 4 passage/query trong CPU pilot. Document scorer kết hợp logit tốt nhất, passage hỗ trợ thứ hai và fusion rank. Evidence giữ đúng text nguồn, kiểm tra cha và substring trước khi xuất.

V1 trong `r2ai/` biểu diễn BM25 bằng title + tối đa 6 chunk đầu, dense document bằng title + 2 chunk đầu, rồi rerank pool document. V2 index trực tiếp passage ở mọi vị trí và cho phép passage đưa document mới vào pool. Production V1 không bị sửa.
""")
    write(reports / "V2_IMPLEMENTATION_REPORT.md", f"""
# Tình trạng triển khai

**Trạng thái:** `{status}`.

- Đã triển khai: remote pinned Parquet sampler, audit mẫu, source windows, document/passage FTS5, dense FAISS pilot, fusion RRF, document diversity, reranker CPU, document scorer, evidence selector, provenance validation, official-schema preview và test.
- Mẫu smoke: {corpus.get('sample_rows', 0)} source chunks, {index.get('sparse', {}).get('passages', 0)} passages, {n_queries} query đã truy hồi; dense index {'đã tạo' if dense_bytes else 'chưa tạo'}; rerank {'đã chạy' if complete else 'chưa hoàn tất'}.
- Mẫu nhiều chunk/document: {extended_corpus.get('sample_rows', 0)} source chunks, {extended_index.get('sparse', {}).get('passages', 0)} passages của {extended_corpus.get('sample_unique_documents', 0)} document; chỉ benchmark lexical, chưa có dense/rerank cho mẫu này.
- Chưa có: index full corpus, official gold/scorer, full-corpus V2 submission, đánh giá F2 tương đương V1, resume/sharded dense build cho 20,5 triệu passage.
""")
    write(reports / "V2_DATA_AUDIT.md", f"""
# Kiểm tra dữ liệu

- HF revision `{corpus.get('source_revision')}`; footer báo {corpus.get('source_rows', 0):,} rows/{corpus.get('source_row_groups', 0)} row groups. Remote revision đã ghim; SHA-256 nguồn `{corpus.get('source_expected_sha256')}` lấy từ manifest audit trước, **chưa băm lại toàn file qua mạng**. SHA-256 pilot local `{corpus.get('sample_sha256')}`.
- Query local {corpus.get('query_rows', 0):,} rows, URL mapping {corpus.get('mapping_rows', 0):,} IDs. Không có ID mẫu thiếu trong mapping: {len(corpus.get('missing_sample_doc_ids', []))}. Duplicate `(doc_id, chunk_index)` trong mẫu smoke: {corpus.get('duplicate_sample_keys', 0)}; text rỗng: {corpus.get('empty_sample_texts', 0)}.
- Mẫu nhiều chunk: {audit.get('rows', 0)} hàng/{audit.get('documents', 0)} documents; title không nhất quán {audit.get('title_inconsistent_documents', 0)} documents; text lặp trong cùng document {audit.get('duplicate_text_within_document', 0)}; {audit.get('short_lt10_words', 0)} chunks dưới 10 từ; {audit.get('documents_with_gt20_chunks', 0)} documents trên 20 chunks; nhiều nhất {audit.get('max_chunks_per_document', 0)} chunks/document.
- Phân bố độ dài p10/p50/p90: {audit.get('word_length_p10_p50_p90', [])} từ. Proxy hệ chữ: {audit.get('script_proxy_counts', {})}; đây không phải nhãn ngôn ngữ.
- Có {extended_corpus.get('sample_late_chunks_gt6', 0)} chunks với `chunk_index > 6` trong mẫu nhiều chunk. Chỉ kiểm tra mẫu 3 row groups [0, 51, 102]; chưa kiểm tra uniqueness và chất lượng toàn bộ 20,5 triệu hàng.
""")
    write(reports / "V2_ENVIRONMENT_REPORT.md", f"""
# Môi trường pilot

- Repo branch `{branch}`, commit `{commit}`; source V1 baseline nằm trong `r2ai/` (`bm25index.py`, `index.py`, `fuse.py`, `rank.py`, `validate.py`). Các thay đổi V2 hiện ở working tree, chưa push.
- Python {env.get('python')}, logical CPUs {env.get('cpu_count')}, RAM tổng {env.get('ram_total_bytes', 0)/1e9:.2f} GB, khả dụng khi lấy mẫu {env.get('ram_available_bytes', 0)/1e9:.2f} GB, disk trống {env.get('disk_free_bytes', 0)/1e9:.1f} GB.
- GPU thấy qua `nvidia-smi`: NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB; pilot chạy CPU theo giới hạn tác vụ.
- Venv đã cần `huggingface-hub>=1.5,<2` để tương thích Transformers cài tại máy. Model cache cục bộ, `HF_HUB_OFFLINE=1` cho bước encode/rerank.
""")
    write(reports / "V2_CHUNKING_REPORT.md", f"""
# Chunking và provenance

Pilot dùng target 192 từ, overlap 15%, ưu tiên kết thúc câu/dòng gần mốc. Chỉ cửa sổ nằm trong **một** source chunk; không nối các chunk khác nhau. Passage lưu `source_doc_id` dưới tên `doc_id`, `source_chunk_index`, `start_char`, `end_char`, `passage_id`, title và text substring. Cửa sổ ngắn giữ nguyên toàn chunk.

Smoke: {corpus.get('sample_rows', 0)} source chunks → {index.get('sparse', {}).get('passages', 0)} passages. Mẫu nhiều chunk/document: {extended_corpus.get('sample_rows', 0)} → {extended_index.get('sparse', {}).get('passages', 0)} passages. Kích thước 256/384 chưa được đo về relevance; cấu hình chỉ ghi chúng như thí nghiệm sau.
""")
    write(reports / "V2_INDEX_REPORT.md", f"""
# Index pilot

- Smoke sparse SQLite FTS5: {sparse_bytes:,} bytes; dense FAISS IndexFlatIP: {dense_bytes:,} bytes, {index.get('dense', {}).get('vectors', 0)} vectors × {index.get('dense', {}).get('dimension', 0)} dimensions. Inner product trên embeddings L2-normalized.
- Mẫu nhiều chunk sparse SQLite: {full_sparse_bytes:,} bytes. Index document chứa **toàn bộ source chunks** dưới dạng postings từng chunk, aggregate về document; passage index tra trực tiếp trên mọi window.
- FAISS ID 0-based + 1 ánh xạ `passages.rowid`, từ đó truy ra `passage_id`, `doc_id`, chunk nguồn và exact text. Index dense hiện là exact **chỉ cho pilot**; không được mở rộng IndexFlatIP này lên 20,5 triệu passage trong RAM 16,85 GB.
- Kiểm tra lại substring và deterministic ID từ source extraction: smoke {grounded_smoke[0]}/{grounded_smoke[1]}, mẫu nhiều chunk {grounded_extended[0]}/{grounded_extended[1]} passages hợp lệ.
- Chưa có benchmark ANN recall/memory nên chưa chọn IVF/HNSW/PQ. Full-scale sparse cũng cần phân shard/resume và đo kích thước thực.
""")
    if (pilot / "V2_RETRIEVAL_BENCHMARK.csv").exists():
        shutil.copyfile(pilot / "V2_RETRIEVAL_BENCHMARK.csv", reports / "V2_RETRIEVAL_BENCHMARK.csv")
    if (pilot / "V2_RETRIEVAL_ABLATION.csv").exists():
        shutil.copyfile(pilot / "V2_RETRIEVAL_ABLATION.csv", reports / "V2_RETRIEVAL_ABLATION.csv")
    with (reports / "V2_INDEX_BUILD_BENCHMARK.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pilot", "index", "rows", "bytes", "build_seconds"])
        writer.writeheader()
        writer.writerows([
            {"pilot": "smoke", "index": "sqlite_fts5", "rows": index.get("sparse", {}).get("passages", 0),
             "bytes": sparse_bytes, "build_seconds": "not_instrumented"},
            {"pilot": "smoke", "index": "faiss_flat_ip", "rows": index.get("dense", {}).get("vectors", 0),
             "bytes": dense_bytes, "build_seconds": "not_instrumented"},
            {"pilot": "full_docs_sample", "index": "sqlite_fts5",
             "rows": extended_index.get("sparse", {}).get("passages", 0),
             "bytes": full_sparse_bytes, "build_seconds": "not_instrumented"},
        ])
    write(reports / "V2_RERANKING_REPORT.md", f"""
# Reranking và document scorer

Model baseline `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, Identity logits, CPU, max length 384 tokens, tối đa 4 candidates/query. Authentic logits được ghi vào `V2_RERANKED_CANDIDATES.parquet`; tokenizer kiểm tra input trước truncation: {truncated}/{total_reranked} cặp vượt giới hạn. Nếu vượt, logit chỉ ứng với input đã truncate.

`V2_DOCUMENT_SCORING.csv` so control max-logit với scorer đơn giản dùng passage thứ hai và fusion rank. Có {multi_support}/{len(doc_score_rows)} document-query có từ hai passage đã rerank; nếu bằng 0 thì pilot chưa đo được tác động của tín hiệu hỗ trợ thứ hai. Chưa có nhãn để quyết định scorer nào tăng relevance; không hiệu chỉnh theo gold. Rerank hiện {'đã chạy' if complete else 'chưa hoàn tất'}.
""")
    write(reports / "V2_EVIDENCE_SELECTION_REPORT.md", """
# Chọn evidence

Từ reranked passages, chọn document theo score tổng hợp, rồi chọn exact source windows có parent trong document output. Loại trùng `(doc_id, chunk_text)`. `assert_grounded` kiểm tra mỗi text là substring của source chunk đúng document.

Chưa thử mở rộng cửa sổ ngoài một source chunk vì extracted Parquet không chứa offset body/HTML gốc. Không có LLM rewrite. Official text/LCS scorer chưa có tại local nên không thể tối ưu F2 từ kết quả pilot.
""")
    write(reports / "V2_EVALUATION_REPORT.md", f"""
# Đánh giá V1/V2

V1 leaderboard đã ghi nhận: FINAL_SCORE 0.1681; DOCS_F2MACRO 0.2802; CHUNKS_F2MACRO 0.0560. V2 chỉ trên pilot {corpus.get('sample_rows', 0)} source chunks, {n_queries} query; không có gold/scorer tương ứng nên **PENDING_OFFICIAL_SCORING**. Không tuyên bố V2 vượt V1.

Technical metrics trên cùng pilot: trung bình {global_docs if global_docs is not None else 'chưa đo'} unique documents ở global top-200 so với {diverse_docs if diverse_docs is not None else 'chưa đo'} ở diverse top-200; retrieval latency trung bình {avg_latency if avg_latency is not None else 'chưa đo'} giây/query (không gồm encode query/rerank). Mỗi nhánh có hit count theo query trong CSV. Các số này không đo relevance.

CPU rerank budget 4/query giới hạn output thực tế tối đa 4 document và 4 evidence/query dù schema cho phép 10. Không dùng tỷ lệ này để ước lượng F2 full-scale.

Lexical A/B trên **cùng mẫu 72 document, 50 query**: V1-style title + 6 chunk đầu so với V2 toàn source chunks, top-20. V2 có trung bình {v2_only if v2_only is not None else 'chưa đo'} document/query không nằm trong V1 top-20; {queries_v2_only}/{len(lexical_rows)} query có ít nhất một document như vậy. Đây chỉ là thay đổi candidate set, chưa xác minh document đó relevant.

Thiếu full-corpus prediction trên cùng 1.200 query, official matching/duplicate logic, và nhãn exhaustively adjudicated. Schema preview có 1.150 query rỗng có chủ ý để test validator; không phải submission thi đấu.
""")
    write(reports / "V2_RESOURCE_ESTIMATE.md", f"""
# Tài nguyên mở rộng

Pilot smoke: {index.get('sparse', {}).get('passages', 0)} passages; sparse {sparse_bytes/1e6:.2f} MB; dense {dense_bytes/1e6:.2f} MB. Mẫu lexical lớn hơn: {extended_index.get('sparse', {}).get('passages', 0)} passages, SQLite {full_sparse_bytes/1e6:.2f} MB. Tỉ lệ pilot SQLite không thể nội suy tuyến tính tin cậy do vocabulary/domain/row-group sampling.

Nếu 20,52 triệu chunks tạo ít nhất một vector 1.024-d float32, riêng vector thô khoảng **84,1 GB** (chưa gồm FAISS, metadata, model và OS); máy có RAM tổng {env.get('ram_total_bytes', 0)/1e9:.2f} GB. Nếu mọi chunk chia thêm passage, chi phí tăng tiếp. Vì vậy cần sharded encode và benchmark ANN recall so exact trên sample trước khi chọn IVF/HNSW/PQ. GPU RTX 4050 6 GB không đủ để giữ full index trong VRAM; chưa có phép đo throughput full-scale để ước lượng thời gian chắc chắn.
""")
    write(reports / "V2_RUNBOOK.md", """
# Lệnh chạy lại trên PowerShell tại repo root

```powershell
$env:HF_HOME = (Join-Path (Get-Location) '.cache/huggingface')
$py = Join-Path (Get-Location) '.venv\\Scripts\\python.exe'
& $py -m scripts.workflows.vibiomir.prepare_v2 --output outputs/retrieval_v2/new_pilot --per-group 64
& $py -m scripts.workflows.vibiomir.audit_v2 --pilot outputs/retrieval_v2/new_pilot
& $py -m scripts.workflows.vibiomir.build_v2_indexes --pilot outputs/retrieval_v2/new_pilot
$env:HF_HUB_OFFLINE = '1'
& $py -m scripts.workflows.vibiomir.build_v2_indexes --pilot outputs/retrieval_v2/new_pilot --dense --batch-size 4
& $py -m scripts.workflows.vibiomir.retrieve_v2 --pilot outputs/retrieval_v2/new_pilot --queries 50 --budget 200
& $py -m scripts.workflows.vibiomir.evaluate_v2 --pilot outputs/retrieval_v2/new_pilot --rerank-budget 4
& $py r2ai/validate.py outputs/retrieval_v2/new_pilot/V2_SCHEMA_PREVIEW_NOT_FOR_SUBMISSION.zip
& $py -m scripts.workflows.vibiomir.report_v2 --pilot outputs/retrieval_v2/new_pilot
```

Mỗi build từ chối ghi đè index/predictions hiện có. Tạo thư mục pilot mới khi chạy lại. Remote sampler chỉ đọc 3 row groups của revision đã ghim. Preview ZIP có record rỗng ngoài pilot và tuyệt đối không dùng để nộp.

Sau khi được phê duyệt GPU job và kiểm tra VRAM, dùng **thư mục pilot mới** và thay ba bước model bằng lệnh dưới đây. Chưa thử các lệnh GPU trong phiên này:

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
& $py -m scripts.workflows.vibiomir.prepare_v2 --output outputs/retrieval_v2/gpu_pilot --per-group 64
& $py -m scripts.workflows.vibiomir.audit_v2 --pilot outputs/retrieval_v2/gpu_pilot
& $py -m scripts.workflows.vibiomir.build_v2_indexes --pilot outputs/retrieval_v2/gpu_pilot
$env:HF_HUB_OFFLINE = '1'
& $py -m scripts.workflows.vibiomir.build_v2_indexes --pilot outputs/retrieval_v2/gpu_pilot --dense --batch-size 4 --device cuda:0
& $py -m scripts.workflows.vibiomir.retrieve_v2 --pilot outputs/retrieval_v2/gpu_pilot --queries 50 --budget 200 --device cuda:0
& $py -m scripts.workflows.vibiomir.evaluate_v2 --pilot outputs/retrieval_v2/gpu_pilot --rerank-budget 4 --device cuda:0
& $py r2ai/validate.py outputs/retrieval_v2/gpu_pilot/V2_SCHEMA_PREVIEW_NOT_FOR_SUBMISSION.zip
```

Các model được nạp trong từng process riêng.
""")
    write(reports / "V2_FINAL_DECISION.md", f"""
# Quyết định

**{status}**. {'Pipeline smoke đã tạo dense, rerank, predictions và schema preview.' if complete else 'Các module/index pilot còn thiếu ít nhất một bước dense/rerank/prediction.'}

Chưa sẵn sàng thay V1: chưa có full-corpus V2 prediction, official F2 cùng phase, ANN recall/memory và resource benchmark full-scale. Bước tiếp: kiểm tra pilot schema/trace, tăng pilot có nhãn dev/holdout, đo candidate coverage, xây sharded sparse/dense trên server sau phê duyệt tài nguyên, rồi nộp chính thức do người dùng quyết định.
""")
    print(f"reports={reports} status={status}")


if __name__ == "__main__":
    main()
