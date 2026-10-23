"""Build the no-edit Colab notebook for a second independent T4 canary."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.v4.run_full_baseline import _write_bytes_checked

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "outputs/v4/p5_3a_colab_canary_20261022/handoff.json"
NOTEBOOK = ROOT / "notebooks/p5_3c_t4_repro.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


def main() -> Path:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    if (handoff.get("format") != "medrag-v4-p5-3a-handoff-v1"
            or handoff.get("pair_count") != 1200
            or handoff.get("query_count") != 6):
        raise ValueError("P5.3A immutable handoff differs")
    bundle_sha = handoff["archive_sha256"]
    cells = [
        markdown("""# P5.3C — Independent Tesla T4 reproducibility run

Chọn **Runtime → Change runtime type → T4 GPU**, bấm **Run All**, cho phép
mount Google Drive và upload `p5_3a_colab_bundle.zip` khi được hỏi. Không sửa
đường dẫn hoặc inference code.

Notebook dùng lại đúng immutable pair bundle và scorer P5.2, nhưng ghi vào
Drive namespace `P5_3C_T4_REPEAT`, tách khỏi lần P5.3A. Bundle không chứa score
cache. Mỗi query phải ghi nhận **200 model calls, 0 cache hits**. Checkpoint chỉ
được dùng để tiếp tục chính repeat run này nếu Colab ngắt.
"""),
        code("""from pathlib import Path
from google.colab import drive

DRIVE_OUTPUT_DIR = Path('/content/drive/MyDrive/medical-rag/P5_3C_T4_REPEAT')
drive.mount('/content/drive')
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print('Independent repeat output:', DRIVE_OUTPUT_DIR)
"""),
        code(f"""import hashlib, os, subprocess, sys, zipfile
from google.colab import files

EXPECTED_BUNDLE_SHA256 = '{bundle_sha}'
uploaded = files.upload()
assert len(uploaded) == 1 and 'p5_3a_colab_bundle.zip' in uploaded, 'Upload đúng p5_3a_colab_bundle.zip'
bundle_path = Path('/content/p5_3a_colab_bundle.zip')
bundle_path.write_bytes(uploaded['p5_3a_colab_bundle.zip'])
assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == EXPECTED_BUNDLE_SHA256
with zipfile.ZipFile(bundle_path) as zf:
    requirements_path = Path('/content/p5_3c_requirements.txt')
    requirements_path.write_bytes(zf.read('p5_3a_requirements.txt'))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'torch==2.6.0', 'torchvision==0.21.0',
                '--index-url', 'https://download.pytorch.org/whl/cu124'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                '-r', str(requirements_path)], check=True)
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

launcher_root = Path('/content/p5_3c_launcher')
if launcher_root.exists():
    shutil.rmtree(launcher_root)
launcher_root.mkdir()
with zipfile.ZipFile(bundle_path) as zf:
    assert zf.testzip() is None
    assert all(not name.startswith('/') and '..' not in Path(name).parts for name in zf.namelist())
    zf.extractall(launcher_root)
launcher = launcher_root / 'scripts/v4/p5_3a_colab_orchestrator.py'
environment = os.environ.copy()
environment['PYTHONPATH'] = str(launcher_root)
environment['HF_HUB_DISABLE_TELEMETRY'] = '1'
subprocess.run([sys.executable, str(launcher), '--bundle', str(bundle_path),
                '--drive-output', str(DRIVE_OUTPUT_DIR),
                '--expected-sha256', EXPECTED_BUNDLE_SHA256],
               env=environment, check=True)
run_dir = DRIVE_OUTPUT_DIR / ('canary_' + EXPECTED_BUNDLE_SHA256[:12])
source = run_dir / 'p5_3a_colab_scores.zip'
target = run_dir / 'p5_3c_t4_repeat_scores.zip'
if target.exists():
    assert hashlib.sha256(target.read_bytes()).hexdigest() == hashlib.sha256(source.read_bytes()).hexdigest()
else:
    shutil.copy2(source, target)
print('Download this file for server verification:', target)
print('Repeat ZIP SHA-256:', hashlib.sha256(target.read_bytes()).hexdigest())
"""),
        markdown("""Đưa `p5_3c_t4_repeat_scores.zip` về server và chạy:

```bash
python -m scripts.v4.verify_p5_3c_t4_reproducibility \
  --repeat-zip outputs/v4/p5_3c_colab_import/p5_3c_t4_repeat_scores.zip
```

Verifier yêu cầu hai run có session provenance độc lập, cùng pinned runtime và
model snapshot, rồi so exact scores, pair ranking, selected objects và BTC
payload. Không dùng P2-A0 mixed-hardware scores làm target của repeat gate.
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
