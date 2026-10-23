"""Build the upload-and-Run-All Colab notebook for full A0_T4_V1 scores."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.v4.run_full_baseline import _write_bytes_checked

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "outputs/v4/a0_t4_v1_colab_bundle_20261022/handoff.json"
NOTEBOOK = ROOT / "notebooks/a0_t4_v1_full_colab.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


def main() -> Path:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    if (handoff.get("format") != "medrag-v4-a0-t4-v1-colab-handoff-v1"
            or handoff.get("query_count") != 1200
            or handoff.get("pair_count") != 240000):
        raise ValueError("Full A0_T4_V1 Colab handoff differs")
    bundle_sha = handoff["archive_sha256"]
    cells = [
        markdown("""# A0_T4_V1 — Full P5 Colab Tesla T4 scoring

Chọn **Runtime → Change runtime type → T4 GPU**, mount Google Drive và bấm
**Run All**. Khi được hỏi, upload đúng một file
`a0_t4_v1_colab_bundle.zip`. Không cần sửa code hoặc đường dẫn.

Bundle chứa 1.200 immutable query pair shards, 240.000 pairs và scorer P5.3A
đã dùng cho canary. Notebook chỉ rerank trên Tesla T4; nó không chạy retrieval.
Mỗi query được checkpoint kèm proof và hashes sau khi ghi nhận 200 model calls,
0 cache hits. Nếu Colab ngắt, chạy lại notebook với cùng bundle và cùng Drive
folder để tiếp tục các query còn thiếu. Khi hoàn tất, tải `a0_t4_v1_scores.zip`
trong Drive về server để import/finalize.
"""),
        code("""from pathlib import Path
from google.colab import drive

DRIVE_OUTPUT_DIR = Path('/content/drive/MyDrive/medical-rag/A0_T4_V1_FULL')
drive.mount('/content/drive')
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print('Durable score checkpoints:', DRIVE_OUTPUT_DIR)
"""),
        code(f"""import hashlib, os, subprocess, sys, zipfile
from google.colab import files

EXPECTED_BUNDLE_SHA256 = '{bundle_sha}'
uploaded = files.upload()
assert len(uploaded) == 1 and 'a0_t4_v1_colab_bundle.zip' in uploaded
bundle_path = Path('/content/a0_t4_v1_colab_bundle.zip')
bundle_path.write_bytes(uploaded['a0_t4_v1_colab_bundle.zip'])
assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == EXPECTED_BUNDLE_SHA256, 'Bundle hash mismatch'
with zipfile.ZipFile(bundle_path) as zf:
    requirements = Path('/content/a0_t4_v1_requirements.txt')
    requirements.write_bytes(zf.read('p5_3a_requirements.txt'))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'torch==2.6.0', 'torchvision==0.21.0',
                '--index-url', 'https://download.pytorch.org/whl/cu124'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(requirements)], check=True)
print('Immutable bundle SHA-256:', EXPECTED_BUNDLE_SHA256)
"""),
        code("""subprocess.run(['nvidia-smi'], check=True)
subprocess.run([sys.executable, '-c',
                "import torch; assert torch.cuda.is_available(); "
                "assert torch.cuda.device_count() == 1; "
                "assert 'T4' in torch.cuda.get_device_name(0); "
                "assert torch.version.cuda == '12.4'; "
                "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"],
               check=True)
"""),
        code("""import shutil

launcher_root = Path('/content/a0_t4_v1_launcher')
if launcher_root.exists():
    shutil.rmtree(launcher_root)
launcher_root.mkdir()
with zipfile.ZipFile(bundle_path) as zf:
    assert zf.testzip() is None
    assert all(not name.startswith('/') and '..' not in Path(name).parts for name in zf.namelist())
    zf.extractall(launcher_root)
launcher = launcher_root / 'scripts/v4/p5_3a_colab_orchestrator.py'
env = os.environ.copy()
env['PYTHONPATH'] = str(launcher_root)
env['HF_HUB_DISABLE_TELEMETRY'] = '1'
subprocess.run([sys.executable, str(launcher), '--bundle', str(bundle_path),
                '--drive-output', str(DRIVE_OUTPUT_DIR),
                '--expected-sha256', EXPECTED_BUNDLE_SHA256], env=env, check=True)
run_dir = DRIVE_OUTPUT_DIR / ('canary_' + EXPECTED_BUNDLE_SHA256[:12])
source = run_dir / 'p5_3a_colab_scores.zip'
target = run_dir / 'a0_t4_v1_scores.zip'
if target.exists():
    assert hashlib.sha256(target.read_bytes()).hexdigest() == hashlib.sha256(source.read_bytes()).hexdigest()
else:
    shutil.copy2(source, target)
print('Download score archive:', target)
print('Score ZIP SHA-256:', hashlib.sha256(target.read_bytes()).hexdigest())
"""),
        markdown("""Import and CPU finalization commands on the server:

```bash
.venv-phase2-batch/bin/python -m scripts.v4.import_a0_t4_v1_colab_scores \\
  --scores-zip outputs/v4/a0_t4_v1_colab_import/a0_t4_v1_scores.zip

.venv-phase2-batch/bin/python -m scripts.v4.run_p5_2_split finalize \\
  --mode full \\
  --config configs/v4_submission_p2_a0.yaml \\
  --output outputs/v4/a0_t4_v1_full_20261022
```

The importer validates pair IDs, hashes, model/runtime provenance and complete
coverage before writing score shards. Finalization uses the frozen CPU selector,
source grounding validator and BTC exporter.
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
