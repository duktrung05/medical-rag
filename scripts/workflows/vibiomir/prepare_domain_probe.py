"""Choose a small deterministic per-domain probe from missing official URLs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.workflows.vibiomir.plan_recovery_batches import BASE, REVISION, SCHEMA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--exclude-domain", action="append", default=[])
    parser.add_argument("--exclude-suffix", action="append", default=[])
    parser.add_argument("--per-domain", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.per_domain <= 10:
        raise ValueError("Probe per-domain size must be 1..10")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    ledger = json.loads((BASE / "ledger_manifest.json").read_text())
    rows = pq.read_table(BASE / "needs_fetch.parquet",
                         columns=["id", "official_url", "hostname"]).to_pylist()
    groups = {}
    for row in rows:
        hostname = row["hostname"]
        if (hostname.endswith(args.suffix) and hostname not in args.exclude_domain
                and not any(hostname.endswith(suffix)
                            for suffix in args.exclude_suffix)):
            groups.setdefault(hostname, []).append(row)
    chosen = []
    for hostname in sorted(groups):
        for round_index, row in enumerate(sorted(groups[hostname], key=lambda value: value["id"])[:args.per_domain]):
            chosen.append({"probe_round": round_index,
                           "id": row["id"], "url": row["official_url"],
                           "domain": hostname,
                           "domain_url_count": ledger["domains"][hostname]["needs_fetch"]})
    # Each batch worker sees alternating domains, rather than a run of one site.
    chosen.sort(key=lambda row: (row["probe_round"], row["domain"], row["id"]))
    for row in chosen:
        del row["probe_round"]
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "input.parquet"
    pq.write_table(pa.Table.from_pylist(chosen, schema=SCHEMA), input_path,
                   compression="zstd")
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    (output / "sample_summary.json").write_text(json.dumps({
        "output_sha256": digest, "corpus_revision": REVISION,
        "rows": len(chosen), "per_domain": args.per_domain,
        "domain_counts": dict(Counter(row["domain"] for row in chosen))},
        indent=2) + "\n")
    print(json.dumps({"rows": len(chosen), "domains": len(groups),
                      "input": str(input_path)}, indent=2))


if __name__ == "__main__":
    main()
