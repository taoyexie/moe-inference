"""Bundle the retained execution chain; preserve the original measured source zip."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
NAME = 'kite-parallel-decode-clean-20261016-v1'
SCRIPTS = ['audit_kite_e2e.py','audit_kite_parallel_decode.py','expert_paths.py',
    'kite_advantage.py','kite_controller.py','kite_e2e.py','kite_e2e_core.py',
    'kite_e2e_runtime.py','kite_e2e_torch.py','kite_parallel_decode.py',
    'kite_readiness.py','kite_trace.py','multi_trace.py','package_kite_parallel_decode.py',
    'plot_kite_parallel_results.py','qwen_smoke.py','screen_kite_parallel_decode.py',
    'workload_baseline.py']
TESTS = ['test_expert_paths.py','test_kite_advantage.py','test_kite_controller.py',
    'test_kite_e2e_core.py','test_kite_e2e_dispatch.py','test_kite_e2e_driver.py',
    'test_kite_e2e_layout.py','test_kite_parallel_decode.py','test_kite_parallel_decode_driver.py',
    'test_kite_parallel_freeze.py','test_kite_performance_scope.py','test_kite_readiness.py',
    'test_kite_trace.py','test_multi_trace.py','test_workload_baseline.py']
FILES = ['README.md','requirements-qwen.txt','requirements-modelscope.txt',
    'docs/CURRENT_PROJECT.md','docs/SERVER_DATA.md','docs/KITE_PARALLEL_DECODE.md'] + [
        'scripts/'+name for name in SCRIPTS] + ['tests/'+name for name in TESTS]


def main():
    target = ROOT/'dist'/(NAME+'.zip')
    target.parent.mkdir(parents=True,exist_ok=True)
    # Validate all inputs before creating an immutable zip; no partial bundle for missing files.
    missing = [name for name in FILES if not (ROOT/name).is_file()]
    if missing:
        raise FileNotFoundError('Missing allowlisted source files: '+', '.join(missing))
    manifest = dict(name=NAME,files=[],
        baseline='existing four-path fixed costs; WiSP-style two-cost baseline not implemented',
        scope='layout cleanup only; inference implementation and measured protocol unchanged')
    with ZipFile(target,'x',ZIP_DEFLATED) as archive:
        for filename in FILES:
            data = (ROOT/filename).read_bytes()
            archive.writestr(NAME+'/'+filename,data)
            manifest['files'].append(dict(path=filename,bytes=len(data),sha256=sha256(data).hexdigest()))
        archive.writestr(NAME+'/bundle_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(zip=str(target),bytes=target.stat().st_size,sha256=sha256(target.read_bytes()).hexdigest()),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
