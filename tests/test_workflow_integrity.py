"""Repository workflow gates after removing unused scaffolds."""
import ast,re
from pathlib import Path
import yaml
from src.config import load_config
from scripts.run_dense_smoke import SmokeConfig


def test_sources_parse_on_supported_python311_and_local_imports_resolve():
    for path in list(Path('src').rglob('*.py'))+list(Path('scripts').glob('*.py')):
        tree=ast.parse(path.read_text(encoding='utf-8-sig'),feature_version=(3,11))
        for node in ast.walk(tree):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            for name in names:
                if name and name.startswith(('src.','scripts.')):
                    target=Path(*name.split('.'))
                    assert target.with_suffix('.py').exists() or target.is_dir(),(str(path),name)


def test_configs_use_their_own_runtime_or_smoke_schema():
    for path in Path('configs').glob('*.yaml'):
        raw=yaml.safe_load(path.read_text(encoding='utf-8'))
        if 'model_name' in raw and 'retrieval' not in raw:SmokeConfig.model_validate(raw)
        else:load_config(path)


def test_readme_local_links_exist():
    text=Path('README.md').read_text(encoding='utf-8-sig')
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if not target.startswith(('http://','https://','#')):
            assert Path(target.split('#')[0]).exists(),target
