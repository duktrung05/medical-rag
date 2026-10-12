# Lệnh chạy lại trên PowerShell tại repo root

```powershell
$env:HF_HOME = (Join-Path (Get-Location) '.cache/huggingface')
$py = Join-Path (Get-Location) '.venv\Scripts\python.exe'
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
