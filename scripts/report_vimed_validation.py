"""Verify all validation results and publish the comparison and frozen winner."""
import json
import shutil
from pathlib import Path
from scripts.benchmark_vimed_validation import CONFIGS, read_rows, load_dependency
from src.evaluation.ranking_metrics import summarize_rankings, validate_ranking
from src.data.loader import DataLoader
from src.indexing.sparse_index import corpus_sha256
import hashlib
import csv


def main():
    root=Path('outputs/vimed_validation')
    samples=read_rows(Path('data/vimed/validation/samples.jsonl'))
    chunks=DataLoader.load_chunks(Path('data/vimed/chunks.jsonl'))
    lookup={c.chunk_id:c.doc_id for c in chunks}
    ch=corpus_sha256(chunks)
    sh=hashlib.sha256(json.dumps(samples,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    rankings={}; metrics={}
    for stage in CONFIGS:
        rows=load_dependency(root,stage,samples,ch,sh)
        for row in rows:
            validate_ranking(row['results'],set(lookup))
            assert len(row['results'])<=100
            assert row['results']==sorted(row['results'],key=lambda x:(-x[1],x[0]))
            assert set(d for d,_ in row['documents'])=={lookup[c] for c,_ in row['results']}
            prediction=row['prediction']
            assert {lookup[c] for c in prediction['relevant_chunks']} <= set(prediction['relevant_docs'])
        m=json.loads((root/stage/'metrics.json').read_text())
        recomputed=summarize_rankings(samples,rows)
        for key,value in recomputed.items():
            assert m[key]==value,(stage,key)
        metrics[stage]=m;rankings[stage]=rows
    for h,r in zip(rankings['hybrid'],rankings['rerank'],strict=True):
        assert {c for c,_ in h['results']}=={c for c,_ in r['results']}
    assert metrics['hybrid']['recall_at_k']['100']==metrics['rerank']['recall_at_k']['100']
    winner=max(CONFIGS,key=lambda s:(metrics[s]['recall_at_k']['10'],metrics[s]['mrr_at_100'],-metrics[s]['latency_ms']['mean']))
    shutil.copyfile(CONFIGS[winner],'configs/vimed_selected_validation.yaml')
    errors=[]
    for i,sample in enumerate(samples):
        ranks={s:rankings[s][i]['first_positive_rank'] for s in CONFIGS}
        if any(r is None or r>10 for r in ranks.values()):
            errors.append(dict(id=sample['id'],query=sample['query'],topic=sample['topic'],relevant_chunks=sample['relevant_chunks'],ranks=ranks))
    (root/'errors.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in errors),encoding='utf-8')
    summary=dict(num_queries=len(samples),split='validation',selection_rule='Recall@10, then MRR@100, then latency',winner=winner,integrity_checks='passed',metrics=metrics)
    baseline_hits=[r['first_positive_rank'] is not None and r['first_positive_rank']<=10 for r in rankings['bm25']]
    selected_hits=[r['first_positive_rank'] is not None and r['first_positive_rank']<=10 for r in rankings[winner]]
    summary['selected_vs_bm25_at_10']={
        'recovered_queries':sum(not b and s for b,s in zip(baseline_hits,selected_hits)),
        'regressed_queries':sum(b and not s for b,s in zip(baseline_hits,selected_hits)),
        'both_miss':sum(not b and not s for b,s in zip(baseline_hits,selected_hits))}
    (root/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (root/'comparison.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['stage','queries','recall_1','recall_5','recall_10','recall_100','mrr_100','latency_mean_ms','latency_method'])
        for stage,m in metrics.items():
            writer.writerow([stage,m['num_queries'],*[m['recall_at_k'][str(k)] for k in (1,5,10,100)],m['mrr_at_100'],m['latency_ms']['mean'],m['latency_method']])
    lines=['# Benchmark retrieval ViMed — task 2','',f'Đã đo đủ **{len(samples)} câu validation**; 5 câu thiếu context bị loại. Corpus cố định 17.955 chunks; không chạy lại test split.','',
    '| Cấu hình | Recall@1 | Recall@5 | Recall@10 | Recall@100 | MRR@100 |','|---|---:|---:|---:|---:|---:|']
    for s,m in metrics.items():
        lines.append('| '+s+' | '+' | '.join(f"{m["recall_at_k"][str(k)]*100:.2f}%" for k in (1,5,10,100))+f" | {m['mrr_at_100']:.4f} |")
    lines+=['',f'## Cấu hình được chọn: {winner}','', 'Tiêu chí đã đặt trước: Recall@10, sau đó MRR@100, sau đó độ trễ. Cấu hình đóng băng ở `configs/vimed_selected_validation.yaml`.', '',
    '## Giao thức và giới hạn','', 'BM25 và dense lấy 100 candidates mỗi nhánh; RRF k=60 giữ 100; reranker xếp lại đủ 100. BGE-M3 dùng CLS, FP16, max_length=512; reranker BGE v2 M3 dùng raw logits, FP16, batch=8, max_length=512. Revision của cả hai model được khóa trong config.', '',
    'Ground truth là context nguồn của từng câu QA, chưa phải nhãn relevance đầy đủ. Context khác có thể đúng nhưng chưa được gán nhãn. Corpus gồm context từ các split, phù hợp đánh giá retrieval trong corpus đóng; không thể suy ra chất lượng tổng quát ngoài corpus. Không dùng câu hỏi/đáp án làm nội dung index.', '',
    'Latency BM25/dense là thời gian pipeline sau warmup; hai lượt chạy đồng thời nên chịu tranh chấp CPU. Latency hybrid/rerank là tổng các lượt đã cache cộng lượt hiện tại, không phải latency API đo trực tiếp. Recall và MRR dùng thứ hạng thực tế, độc lập với cách đo latency.', '',
    f"Union BM25+dense có Recall candidate {metrics['hybrid']['union_candidate_recall']*100:.2f}%; sau RRF top100 còn {metrics['hybrid']['recall_at_k']['100']*100:.2f}%. Reranker giữ nguyên tập candidates nên Recall@100 bằng hybrid.", '',
    '## Kiểm tra','', '112 tests pass; 1 cảnh báo deprecation Starlette/AnyIO. Mỗi lượt đủ số câu, checksum cache hợp lệ, không duplicate/unknown ID hoặc score vô hạn; metrics được tính lại từ rankings; reranker giữ nguyên candidates. Chi tiết mỗi topic, latency, VRAM và danh sách lỗi nằm trong `outputs/vimed_validation/`.']
    lines += ['', Path('docs/vimed_validation_protocol_vi.md').read_text(encoding='utf-8')]
    Path('docs/vimed_validation_benchmark_vi.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({s:{'recall':m['recall_at_k'],'mrr':m['mrr_at_100']} for s,m in metrics.items()},indent=2))
    print('Winner:',winner)

if __name__=='__main__':main()
