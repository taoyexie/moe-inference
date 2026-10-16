"""Allowlisted baseline/controller source bundle. Never include results, models, or keys."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
NAME = 'kite-workload-baseline-20261015-v1'
FILES = ['requirements-qwen.txt','scripts/kite_trace.py','scripts/qwen_smoke.py',
         'scripts/multi_trace.py','scripts/workload_baseline.py','scripts/kite_controller.py',
         'tests/test_kite_trace.py','tests/test_multi_trace.py','tests/test_workload_baseline.py',
         'tests/test_kite_controller.py','docs/WORKLOAD_BASELINE_GUIDE.md','docs/KITE_RUNTIME_STATUS.md']


def main():
    target = ROOT/'dist'/(NAME+'.zip')
    target.parent.mkdir(parents=True,exist_ok=True)
    manifest = {'name':NAME,'files':[]}
    with ZipFile(target,'x',ZIP_DEFLATED) as archive:
        for name in FILES:
            data = (ROOT/name).read_bytes()
            archive.writestr(NAME+'/'+name,data)
            manifest['files'].append({'path':name,'bytes':len(data),'sha256':sha256(data).hexdigest()})
        archive.writestr(NAME+'/bundle_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'zip':str(target),'bytes':target.stat().st_size,
                     'sha256':sha256(target.read_bytes()).hexdigest()},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
