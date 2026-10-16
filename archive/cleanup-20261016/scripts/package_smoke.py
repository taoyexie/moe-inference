"""Package only source needed by the remote smoke test; never include secrets/results."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
NAME = 'kite-qwen-smoke-20261015-v3'
FILES = ['requirements-qwen.txt', 'requirements-modelscope.txt', 'scripts/kite_trace.py', 'scripts/qwen_smoke.py',
         'tests/test_kite_trace.py', 'docs/QWEN_SMOKE_GUIDE.md']
target = ROOT / 'dist' / (NAME + '.zip')
target.parent.mkdir(parents=True, exist_ok=True)
manifest = {'name': NAME, 'files': []}
with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
    for relative in FILES:
        data = (ROOT/relative).read_bytes()
        archive.writestr(NAME+'/'+relative, data)
        manifest['files'].append({'path': relative, 'sha256': sha256(data).hexdigest(), 'bytes': len(data)})
    archive.writestr(NAME+'/bundle_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
print(json.dumps({'zip': str(target), 'bytes': target.stat().st_size,
                  'remote_folder': '/workspace/'+NAME}, ensure_ascii=False, indent=2))
