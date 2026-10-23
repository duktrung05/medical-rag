"""Small CPU smoke over one preserved ViBioMIR HTML source."""
from __future__ import annotations

import json
from pathlib import Path

from src.medrag.chunking.passage_builder import build_passages
from src.medrag.core.pipeline import Pipeline
from src.medrag.core.paths import project_root
from src.medrag.data.loader import load_crawled_source


def run(query: str = "Bệnh viện Nhân Ái điều trị bệnh nhân HIV/AIDS", max_tokens: int = 256) -> dict:
    root = project_root()
    # These local ID/URL pairs are documented in archive/docs_pre_v4/vibiomir_probe_vi.md.
    # Production must verify each pair against the selected official Parquet.
    mapping = {
        583: "http://special.vietnamplus.vn/benh_vien_nhan_ai",
        565962: "https://suckhoeviet.org.vn/ha-giang-theo-duoi-muc-tieu-tro-thanh-vung-trong-diem-quoc-gia-ve-duoc-lieu-8019.html",
        836542: "http://fk.99.com.cn/news/410997.html",
    }
    sources = [load_crawled_source(version="legacy-local-20261006", official_id=doc_id,
                                   url=url, mapping=mapping, pages_root=root / "data/vibiomir/crawl/pages")
               for doc_id, url in mapping.items()]
    passages = [passage for source in sources for passage in build_passages(source, max_tokens=max_tokens)]
    pipeline = Pipeline(sources, passages)
    return {"sources": {s.official_id: {"path": s.origin, "raw_sha256": s.raw_sha256} for s in sources},
            "passage_count": len(passages), "results": pipeline.retrieve(query, 3),
            "zh_results": pipeline.retrieve("女人经期不能做什么", 3)}


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2))

