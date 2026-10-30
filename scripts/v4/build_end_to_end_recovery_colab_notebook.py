"""Build a Colab launcher for the single full evidence recovery candidate."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "notebooks/v4_end_to_end_evidence_recovery_colab.ipynb"


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": value}


def code(value: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": value}


def main() -> Path:
    cells = [
        markdown("""# V4 evidence recovery — one candidate

Select a GPU runtime in **Runtime → Change runtime type** and run all cells.
**Before Run All:** this notebook file alone does not contain the project data.
Use the flat MyDrive/medical-rag folder shown in the upload screenshot.
It needs pyproject.toml, README.md, src/, scripts/, configs/, and these inputs:

- sparse_full_20261016/ — 28 GB;
- dense_full_gpu3_20261019/ — 12 GB;
- manifest.json, submission.zip, policy_lock.json, query.parquet.

The notebook downloads pinned model snapshots itself. Resume by running all
cells again with the same GPU. The final ZIP appears in Drive at
MyDrive/medical-rag/end_to_end_recovery_20261026/export/submission.zip.

This run does not require A6000 logits or compare them with the current GPU.
The notebook checks file identity, source grounding, caps and ZIP structure.
"""),
        code("""from google.colab import drive
from pathlib import Path
import os, shutil, subprocess, sys

drive.mount('/content/drive')
DRIVE_ROOT = Path('/content/drive/MyDrive/medical-rag')
required = ['pyproject.toml', 'README.md', 'src', 'scripts', 'configs',
            'sparse_full_20261016', 'dense_full_gpu3_20261019',
            'manifest.json', 'submission.zip', 'policy_lock.json', 'query.parquet']
missing = [name for name in required if not (DRIVE_ROOT / name).exists()]
assert not missing, f'Upload has not finished or is missing: {missing}'
runner_path = DRIVE_ROOT / 'scripts/v4/run_end_to_end_evidence_recovery.py'
assert runner_path.is_file() and 'with zipfile.ZipFile(cap_zip)' in runner_path.read_text(), (
    f'Upload the latest runner file: {runner_path}')

# Keep code on Colab local disk; read the large immutable index from Drive.
ROOT = Path('/content/medical-rag-run')
ROOT.mkdir(parents=True, exist_ok=True)
for name in ('src', 'scripts', 'configs'):
    shutil.copytree(DRIVE_ROOT / name, ROOT / name, dirs_exist_ok=True)
for name in ('pyproject.toml', 'README.md'):
    shutil.copy2(DRIVE_ROOT / name, ROOT / name)

def link_file(link, target):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() != target.resolve():
        link.unlink()
    if not link.exists():
        link.symlink_to(target)

index = ROOT / 'outputs/v4/task4/full_index_gpu3_20261019'
index.mkdir(parents=True, exist_ok=True)
link_file(index / 'manifest.json', DRIVE_ROOT / 'manifest.json')
link_file(ROOT / 'outputs/vibiomir_official_verify_20261012/query.parquet',
          DRIVE_ROOT / 'query.parquet')
link_file(ROOT / 'outputs/v4/evidence_window_cap10_full_20261025/export/submission.zip',
          DRIVE_ROOT / 'submission.zip')
OUTPUT_DRIVE = DRIVE_ROOT / 'end_to_end_recovery_20261026'
OUTPUT_DRIVE.mkdir(exist_ok=True)
policy_copy = OUTPUT_DRIVE / 'policy_lock.json'
if not policy_copy.exists():
    shutil.copy2(DRIVE_ROOT / 'policy_lock.json', policy_copy)
assert policy_copy.read_bytes() == (DRIVE_ROOT / 'policy_lock.json').read_bytes()
link_file(ROOT / 'outputs/v4/end_to_end_recovery_20261026', OUTPUT_DRIVE)

os.chdir(ROOT)
os.environ['PYTHONPATH'] = str(ROOT)
os.environ['HF_HOME'] = str(DRIVE_ROOT / '.cache/huggingface')
print('Colab code:', ROOT, 'Drive inputs:', DRIVE_ROOT, 'Drive output:', OUTPUT_DRIVE)
"""),
        code("""# Connect the full index backing files without copying 40 GB again.
sparse = DRIVE_ROOT / 'sparse_full_20261016'
dense = DRIVE_ROOT / 'dense_full_gpu3_20261019'
targets = {'documents.sqlite': sparse / 'documents.sqlite',
           'passages.sqlite': sparse / 'passages.sqlite',
           'corpus.sqlite': sparse / 'corpus.sqlite',
           'lock.json': sparse / 'lock.json',
           'dense': dense / 'dense'}
for name, target in targets.items():
    assert target.exists(), f'Missing full index backing file: {target}'
    link_file(index / name, target)
    assert (index / name).exists()
assert (ROOT / 'outputs/v4/end_to_end_recovery_20261026/policy_lock.json').is_file()
assert (ROOT / 'outputs/v4/evidence_window_cap10_full_20261025/export/submission.zip').is_file()
print('Index, baseline and policy lock present')
"""),
        code("""subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'torch==2.6.0', 'torchvision==0.21.0',
                '--index-url', 'https://download.pytorch.org/whl/cu124'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.',
                'sentence-transformers==4.1.0', 'transformers==4.51.3',
                'tokenizers==0.21.4', 'huggingface-hub==0.36.2'], check=True)
subprocess.run(['nvidia-smi'], check=True)
subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available(); '
                'print(torch.__version__, torch.cuda.get_device_name(0))'], check=True)
"""),
        code("""# Fetch exact pinned revisions if they are not already in the Drive cache.
from huggingface_hub import snapshot_download
for model, revision in [
    ('BAAI/bge-m3', '5617a9f61b028005a4858fdac845db406aefb181'),
    ('BAAI/bge-reranker-v2-m3', '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e')]:
    snapshot_download(repo_id=model, revision=revision)
print('Pinned model snapshots present')
"""),
        code("""# Each stage resumes from immutable per-query shards. A disconnect can be retried.
runner = ['-m', 'scripts.v4.run_end_to_end_evidence_recovery']
for stage in ('retrieve', 'score', 'export'):
    print('Starting stage:', stage, flush=True)
    subprocess.run([sys.executable, *runner, stage], cwd=ROOT,
                   env=os.environ.copy(), check=True)
zip_path = OUTPUT_DRIVE / 'export/submission.zip'
print('Validated candidate ZIP:', zip_path, 'bytes:', zip_path.stat().st_size)
"""),
    ]
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    if TARGET.exists() and TARGET.read_text() != payload:
        raise FileExistsError(f"Refusing to overwrite changed notebook: {TARGET}")
    if not TARGET.exists():
        TARGET.write_text(payload)
    return TARGET


if __name__ == "__main__":
    print(main())
