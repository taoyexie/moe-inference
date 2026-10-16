"""Clarify the measured baseline without changing evidence, figures, or execution code."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.oxml.ns import qn


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/KITE专利技术交底书_终稿.docx'
OUTPUT = ROOT / 'reports/KITE专利技术交底书_终稿_baseline说明版.docx'
QA = ROOT / 'tmp/patent_baseline_20261016'

BASELINE_TEXT = [
    '本次负载共 16 个计算批次：首批包含四条 Prefill，后续 15 批包含四条 Decode。'
    '比较固定成本 baseline 与完整 KITE 两种配置。图例中的 Fixed-cost baseline '
    '指本实施例的固定路径成本、历史曲线分配对照，不是固定显存比例的配置，也不是原始 WiSP 的复现。',
    'baseline 的成本模型：每次完整工作负载均按相同协议独立执行启动标定，而不直接复用另一组的计时值。'
    '对同一采样点的正式计时取中位数，再对不同采样点等权求均值，分别得到 Expert Upload、Expert CPU、'
    'KV Restore 和 KV 前缀重放四个常数。baseline 在标定范围内使用这些常数，不根据 routed-token 数 q '
    '或上下文长度改变 CPU 计算、前缀重放的预测值；没有 CPU FP32 副本时仍额外计入冷转换开销，'
    '超出标定范围仍保守采用 Upload 或 Restore。',
    'baseline 的分配方式：维护与 KITE 相同的历史复用距离统计，估计各候选容量下的 miss，'
    '结合上述固定路径成本计算恢复代价，选择评分最低的可行 Expert/KV 划分。历史建议每 16 批刷新一次，'
    '但不加入当前批次 P/D 请求比例修正，其当前信号融合权重为 0。因此 baseline 可以随历史变化调整容量，'
    '并非人为固定为一半显存；本次运行保持半池附近，是实际历史评分选择的结果。',
    '完整 KITE 在相同执行框架上，将 Expert CPU 和 KV 前缀重放改为带截距的线性路径模型，'
    'Upload 和 Restore 仍使用常数，并以 0.65 权重融合当前 P/D 信号后选取最近可行划分。'
    '两组使用相同的执行器、请求、预算、精度、候选集合和初始空 GPU 缓存，均保留四条 fallback 路径，'
    '按各自模型选择可用路径，均执行相同的冷转换及超出标定范围保护。比较结果用于说明完整策略的整体效果，'
    '不将差异单独归因于阶段修正或某一成本模型。',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def media(path):
    with ZipFile(path) as archive:
        return {n: hashlib.sha256(archive.read(n)).hexdigest()
                for n in archive.namelist() if n.startswith('word/media/')}


def rewrite(paragraph, text):
    formatting = deepcopy(paragraph.runs[0]._r.rPr) if paragraph.runs else None
    for child in list(paragraph._p):
        if child.tag != qn('w:pPr'):
            paragraph._p.remove(child)
    run = paragraph.add_run(text)
    if formatting is not None:
        run._r.insert(0, formatting)


def main():
    original_hash = digest(SOURCE)
    doc = Document(SOURCE)
    target = [p for p in doc.paragraphs if p.text.startswith('本次负载共 16 个计算批次：')]
    assert len(target) == 1, 'Expected one comparison-method paragraph.'
    paragraph = target[0]
    paragraph_format = deepcopy(paragraph._p.pPr)
    rewrite(paragraph, BASELINE_TEXT[-1])
    for text in BASELINE_TEXT[:-1]:
        new = paragraph.insert_paragraph_before()
        if paragraph_format is not None:
            new._p.insert(0, deepcopy(paragraph_format))
        # The replacement paragraph preserves the source font properties.
        run = new.add_run(text)
        run._r.insert(0, deepcopy(paragraph.runs[0]._r.rPr))

    replacements = 0
    for part in doc.part.package.parts:
        element = getattr(part, 'element', None)
        if element is None:
            continue
        for node in element.iter(qn('w:t')):
            if node.text and '基线' in node.text:
                replacements += node.text.count('基线')
                node.text = node.text.replace('基线', 'baseline')
    assert replacements >= 5
    doc.save(OUTPUT)

    checked = Document(OUTPUT)
    with ZipFile(OUTPUT) as archive:
        assert not any('基线' in archive.read(n).decode('utf-8')
                       for n in archive.namelist() if n.startswith('word/') and n.endswith('.xml'))
    assert len(checked.tables) == len(Document(SOURCE).tables) == 4
    assert len(checked.inline_shapes) == len(Document(SOURCE).inline_shapes) == 5
    assert media(SOURCE) == media(OUTPUT), 'Embedded figures must stay byte-identical.'
    assert digest(SOURCE) == original_hash, 'Original final must remain unchanged.'
    assert [[c.text.replace('基线', 'baseline') for c in row.cells]
            for row in Document(SOURCE).tables[3].rows] == [
                [c.text for c in row.cells] for row in checked.tables[3].rows]
    QA.mkdir(parents=True, exist_ok=True)
    manifest = dict(source=str(SOURCE), source_sha256=original_hash, output=str(OUTPUT),
                    output_sha256=digest(OUTPUT), terminology_replacements=replacements,
                    added_paragraphs=len(BASELINE_TEXT)-1, tables=4, images=5,
                    figures_byte_identical=True, result_table_values_unchanged=True,
                    baseline_checked_against=[
                        'scripts/kite_e2e_core.py:RuntimeCosts',
                        'scripts/kite_controller.py:History.score and Controller',
                        'scripts/kite_e2e_runtime.py:Runtime.calibrate',
                        'scripts/kite_advantage.py:OPTIONS'],
                    visual_review='pending')
    (QA/'build_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
