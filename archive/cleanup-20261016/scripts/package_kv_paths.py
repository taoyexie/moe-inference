"""Build a source-only, allowlisted KV benchmark bundle; never include weights or results."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
NAME = 'kite-kv-paths-20261015-v1'
FILES = ['requirements-qwen.txt', 'scripts/kite_trace.py', 'scripts/qwen_smoke.py',
         'scripts/expert_paths.py', 'scripts/kv_paths.py', 'tests/test_kv_paths.py',
         'docs/KV_PATHS_GUIDE.md']


def main():
    target = ROOT/'dist'/(NAME+'.zip')
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'name': NAME, 'files': []}
    with ZipFile(target, 'x', ZIP_DEFLATED) as archive:
        for relative in FILES:
            data = (ROOT/relative).read_bytes()
            archive.writestr(NAME+'/'+relative, data)
            manifest['files'].append({'path': relative, 'sha256': sha256(data).hexdigest(), 'bytes': len(data)})
        archive.writestr(NAME+'/bundle_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({'zip': str(target), 'bytes': target.stat().st_size,
                      'sha256': sha256(target.read_bytes()).hexdigest()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
