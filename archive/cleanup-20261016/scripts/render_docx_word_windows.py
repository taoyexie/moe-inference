"""Canonical render_docx rasterizer with native Word PDF export on Windows.

The bundled renderer has no Windows LibreOffice binary. Native Word is used
read-only for conversion; the source is never saved. No desktop LO fallback.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile

RUNTIME = Path('C:/Users/eyess/.cache/codex-runtimes/codex-primary-runtime/dependencies')
RENDERER = Path('C:/Users/eyess/.codex/plugins/cache/openai-primary-runtime/documents/26.1013.11914/skills/documents/render_docx.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--dpi',type=int,default=130)
    args = parser.parse_args()
    source,output = args.source.resolve(),args.output_dir.resolve()
    task_temp = output.parent/'render_temp'
    task_temp.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir = str(task_temp)
    os.environ['PATH'] = str(RUNTIME/'native/poppler/Library/bin')+os.pathsep+os.environ.get('PATH','')
    module_spec = importlib.util.spec_from_file_location('canonical_docx_renderer',RENDERER)
    renderer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(renderer)
    def word_convert(doc_path,user_profile,convert_tmp_dir,stem,verbose):
        pdf = Path(convert_tmp_dir)/(stem+'.pdf')
        command = [str(RUNTIME/'native/powershell/pwsh.exe'),'-NoLogo','-NoProfile','-STA','-File',
            str(Path(__file__).with_name('render_word_pdf.ps1')),'-Source',doc_path,'-Target',str(pdf)]
        result = subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=100)
        log = result.stdout+'\n'+result.stderr
        print(log,flush=True)
        if result.returncode or not pdf.is_file() or not pdf.stat().st_size:
            raise RuntimeError('Native Word PDF export failed: '+log)
        return str(pdf),log
    renderer.convert_to_pdf = word_convert
    pages = renderer.rasterize(str(source),str(output),args.dpi,verbose=True,emit_pdf=True)
    print('Rendered pages:',len(pages),flush=True)


if __name__=='__main__':
    main()
