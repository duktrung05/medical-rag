"""Compare frozen refinement arms on incomplete known-positive diagnostics."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.v4.diagnose_a0_quality import ROOT, BASE, load_labels, normalize, partition
from scripts.v4.refine_a0_submission import OUTPUT
from src.medrag.evidence.refinement import filter_documents

REPORTS = ROOT / 'outputs/v4/a0_quality_comparison_20261024'


def main():
    REPORTS.mkdir(parents=True, exist_ok=False)
    labels, inputs = load_labels()
    counts = defaultdict(Counter)
    examples = []
    per_query = []
    for qid in range(1, 1201):
        original = json.loads((BASE / 'final' / f'{qid}.json').read_text())
        refined = json.loads((OUTPUT / 'final' / f'{qid}.json').read_text())
        part = partition(qid)
        variants = {
            'baseline': (original['selected'], None),
            'docs_gate_only': (filter_documents(original['selected'], -4.), None),
            'context_only': (original['selected'], refined['original_expanded_evidence']),
            'baseline_refined': (filter_documents(original['selected'], -4.), refined['original_expanded_evidence']),
            'rerank_only': (refined['selected'], None),
            'rerank_refined': (filter_documents(refined['selected'], -4.), refined['expanded_evidence']),
        }
        predictions = {}
        local = defaultdict(Counter)
        for name, (docs, expansions) in variants.items():
            predictions[name] = {d['official_id']: [expansions[str(d['official_id'])]['text']]
                                 if expansions is not None and d['evidence']
                                 else [e['text'] for e in d['evidence']] for d in docs}
            counts[part, name]['queries'] += 1
            counts[part, name]['documents'] += len(docs)
            counts[part, name]['chunks'] += sum(bool(texts) for texts in predictions[name].values())
        for label in labels[qid]:
            if (label['query_text'] is not None and hashlib.sha256(label['query_text'].encode()).hexdigest()
                    != original['query_sha256']):
                raise ValueError('Diagnostic label query identity mismatch')
            parent, needle, group = label['doc_id'], normalize(label['text']), label['label_set']
            outcomes = {}
            for name, mapping in predictions.items():
                result = {'document': parent in mapping, 'chunk_parent': bool(mapping.get(parent)),
                          'exact_anchor': any(needle in normalize(t) for t in mapping.get(parent, []))}
                outcomes[name] = result
                counts[part, name][group + '/labels'] += 1
                for key, value in result.items():
                    counts[part, name][group + '/' + key] += value
                    local[name][group + '/' + key] += value
            if outcomes['baseline'] != outcomes['rerank_refined']:
                examples.append({'query_id': qid, 'partition': part, 'label_set': group, 'doc_id': parent,
                                 **{f'{arm}/{metric}': value for arm in ('baseline', 'rerank_refined')
                                    for metric, value in outcomes[arm].items()}})
        for name, metrics in local.items():
            per_query.append({'query_id': qid, 'partition': part, 'variant': name, **metrics})
        if qid % 100 == 0:
            print(f'Compared {qid}/1200 queries across six fixed variants', flush=True)
    report = {'scope': 'Known positives only; document survival and exact historical/manual span containment, not official F2',
              'inputs_sha256': inputs, 'counts': {f'{part}/{name}': dict(c) for (part, name), c in counts.items()},
              'validation_caveat': 'Query-disjoint split; these historical labels were previously used diagnostically'}
    for filename, rows in [('changed_positive_paths.csv', examples), ('per_query.csv', per_query)]:
        with (REPORTS / filename).open('w', newline='') as stream:
            columns = list(dict.fromkeys(k for row in rows for k in row))
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    (REPORTS / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
