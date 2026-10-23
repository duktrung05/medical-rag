# Scripts

Các script được chia theo luồng để Explorer và lệnh chạy phản ánh cùng một cấu trúc:

```text
scripts/
├── workflows/
│   ├── vibiomir/       dataset chính thức: inventory, crawl, extract, chunk
│   ├── vimedqa/        benchmark lịch sử/regression
│   └── medquad/        regression tiếng Anh
├── experiments/vimedqa/ ablation, calibration, replay, error analysis
├── runtime/            build index, retrieve, evaluate và smoke
└── tooling/            submission, dependency và metadata helpers
```

Ví dụ:

```powershell
python -m scripts.workflows.vibiomir.prepare_vibiomir
python -m scripts.workflows.vibiomir.sample_vibiomir_urls
python -m scripts.workflows.vimedqa.benchmark_vimed_validation --stage bm25
python -m scripts.runtime.retrieve --config configs/vimed_selected_validation.yaml
```

ViBioMIR là luồng dữ liệu chính thức; ViMedQA và MedQuAD phục vụ regression/đối chứng.