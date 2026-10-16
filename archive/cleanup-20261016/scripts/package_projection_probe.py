"""Source-only first-layer probe. Formal v2 files remain unchanged."""
from hashlib import sha256
import json
from zipfile import ZipFile,ZIP_DEFLATED
from package_kite_advantage import FILES,ROOT


def main():
    name = 'kite-projection-probe-20261016-v1'
    target = ROOT/'dist'/(name+'.zip')
    files = FILES+['scripts/first_layer_probe.py','tests/test_first_layer_probe.py',
                   'docs/KITE_PROJECTION_PROBE.md']
    manifest = dict(name=name,files=[])
    with ZipFile(target,'x',ZIP_DEFLATED) as archive:
        for filename in files:
            data = (ROOT/filename).read_bytes()
            archive.writestr(name+'/'+filename,data)
            manifest['files'].append(dict(path=filename,bytes=len(data),sha256=sha256(data).hexdigest()))
        archive.writestr(name+'/bundle_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(zip=str(target),bytes=target.stat().st_size,
        sha256=sha256(target.read_bytes()).hexdigest()),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
