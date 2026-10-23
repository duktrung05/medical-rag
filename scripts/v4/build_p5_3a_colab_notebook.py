"""Render the no-edit Google Colab T4 notebook for the pinned P5.3A bundle."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.v4.run_full_baseline import _write_bytes_checked

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "outputs/v4/p5_3a_colab_canary_20261022/handoff.json"
NOTEBOOK = ROOT / "notebooks/p5_3a_colab.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


def main() -> Path:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    if (handoff.get("format") != "medrag-v4-p5-3a-handoff-v1"
            or handoff.get("archive") != "p5_3a_colab_bundle.zip"
            or handoff.get("pair_count") != 1200):
        raise ValueError("P5.3A bundle handoff differs")
    bundle_sha = handoff["archive_sha256"]
    cells = [
        markdown("""# P5.3A — Google Colab Tesla T4 canary

**Thao tác:** chọn **Runtime → Change runtime type → T4 GPU**, bấm **Run All**,
cho phép mount Google Drive và upload đúng `p5_3a_colab_bundle.zip` khi Colab
hỏi. Không cần sửa đường dẫn hay code. Kết quả được lưu tự động trong
`MyDrive/medical-rag/P5_3A_T4/canary_<bundle hash>/p5_3a_colab_scores.zip`.

Notebook dùng chính hàm `score()` của P5.2. Bundle không có cached scores.
Mỗi query chỉ được checkpoint lên Drive sau khi log xác nhận **200 model calls,
0 cache hits** và score shard qua kiểm tra hash. Nếu Colab ngắt, chạy lại
notebook với cùng bundle; sáu shard đã checkpoint hợp lệ sẽ được giữ, các query
còn thiếu sẽ chấm mới. Tốc độ và ETA chỉ tính cho GPU reranking của canary,
không bao gồm CPU retrieval hoặc full submission.
"""),
        code("""from pathlib import Path
from google.colab import drive

# Cấu hình Google Drive. Giá trị mặc định chạy ngay, không cần sửa.
DRIVE_OUTPUT_DIR = Path('/content/drive/MyDrive/medical-rag/P5_3A_T4')
drive.mount('/content/drive')
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print('Drive checkpoints/output:', DRIVE_OUTPUT_DIR)
"""),
        code(f"""import hashlib, os, subprocess, sys, zipfile
from google.colab import files

EXPECTED_BUNDLE_SHA256 = '{bundle_sha}'
uploaded = files.upload()
assert len(uploaded) == 1 and 'p5_3a_colab_bundle.zip' in uploaded, 'Upload đúng một file p5_3a_colab_bundle.zip'
bundle_path = Path('/content/p5_3a_colab_bundle.zip')
bundle_path.write_bytes(uploaded['p5_3a_colab_bundle.zip'])
assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == EXPECTED_BUNDLE_SHA256, 'Bundle SHA-256 mismatch'
with zipfile.ZipFile(bundle_path) as zf:
    requirements = zf.read('p5_3a_requirements.txt')
requirements_path = Path('/content/p5_3a_requirements.txt')
requirements_path.write_bytes(requirements)

# Pin CUDA/PyTorch và đúng các dependency dùng trong P2-A0. Process scorer
# được mở mới sau pip install để tránh module cũ đã nạp trong kernel.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'torch==2.6.0', 'torchvision==0.21.0',
                '--index-url', 'https://download.pytorch.org/whl/cu124'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                '-r', str(requirements_path)], check=True)
print('Bundle SHA-256:', EXPECTED_BUNDLE_SHA256)
"""),
        code("""# Kiểm tra GPU/CUDA sau khi cài đúng runtime đã khóa.
subprocess.run(['nvidia-smi'], check=True)
subprocess.run([sys.executable, '-c',
                "import torch; assert torch.cuda.is_available(); "
                "assert torch.cuda.device_count() == 1; "
                "assert 'T4' in torch.cuda.get_device_name(0); "
                "assert torch.version.cuda == '12.4'; "
                "print('CUDA:', torch.version.cuda, 'GPU:', torch.cuda.get_device_name(0))"],
               check=True)
"""),
        code("""import shutil

launcher_root = Path('/content/p5_3a_launcher')
if launcher_root.exists():
    shutil.rmtree(launcher_root)
launcher_root.mkdir()
with zipfile.ZipFile(bundle_path) as zf:
    assert zf.testzip() is None, 'Bundle CRC mismatch'
    assert all(not name.startswith('/') and '..' not in Path(name).parts for name in zf.namelist())
    zf.extractall(launcher_root)
launcher = launcher_root / 'scripts/v4/p5_3a_colab_orchestrator.py'
assert launcher.is_file()
environment = os.environ.copy()
environment['PYTHONPATH'] = str(launcher_root)
environment['HF_HUB_DISABLE_TELEMETRY'] = '1'
subprocess.run([sys.executable, str(launcher),
                '--bundle', str(bundle_path),
                '--drive-output', str(DRIVE_OUTPUT_DIR),
                '--expected-sha256', EXPECTED_BUNDLE_SHA256],
               env=environment, check=True)
print('Đã lưu score shards, score manifest, proofs, benchmark và ZIP vào:', DRIVE_OUTPUT_DIR)
"""),
        markdown("""Sau khi Run All hoàn tất, lấy `p5_3a_colab_scores.zip` trong thư
mục Drive được in ở cell cuối và đưa về repository để chạy
`scripts/v4/verify_p5_3a_colab_canary.py --scores-zip <đường dẫn>`.
GO/NO-GO chỉ được quyết định sau bước import và so parity trên CPU.
"""),
    ]
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_checked(NOTEBOOK, (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode())
    return NOTEBOOK


if __name__ == "__main__":
    print(main())
