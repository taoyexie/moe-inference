"""Source-only KITE full-model E2E bundle; no checkpoints, results or credentials."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
NAME = 'kite-e2e-20261016-v2'
FILES = ['requirements-qwen.txt','scripts/kite_trace.py','scripts/qwen_smoke.py',
    'scripts/expert_paths.py','scripts/multi_trace.py','scripts/workload_baseline.py',
    'scripts/kite_controller.py','scripts/kite_e2e_core.py','scripts/kite_e2e_torch.py',
    'scripts/kite_e2e_runtime.py','scripts/kite_e2e.py','scripts/audit_kite_e2e.py',
    'tests/test_kite_e2e_core.py','tests/test_kite_e2e_layout.py','tests/test_kite_e2e_driver.py',
    'tests/test_kite_e2e_dispatch.py',
    'tests/test_kite_controller.py','tests/test_kite_trace.py','tests/test_multi_trace.py',
    'tests/test_workload_baseline.py','docs/KITE_E2E_GUIDE.md']


def main():
    target = ROOT/'dist'/(NAME+'.zip')
    target.parent.mkdir(parents=True,exist_ok=True)
    manifest = dict(name=NAME,files=[])
    with ZipFile(target,'x',ZIP_DEFLATED) as archive:
        for name in FILES:
            data = (ROOT/name).read_bytes()
            archive.writestr(NAME+'/'+name,data)
            manifest['files'].append(dict(path=name,bytes=len(data),sha256=sha256(data).hexdigest()))
        archive.writestr(NAME+'/bundle_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(zip=str(target),bytes=target.stat().st_size,
                          sha256=sha256(target.read_bytes()).hexdigest()),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
