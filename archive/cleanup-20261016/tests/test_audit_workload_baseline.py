"""Regression checks using downloaded records; no GPU or model dependency."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from audit_workload_baseline import audit

SOURCE = ROOT/'results'/'qwen-20261016T003757Z-6ad1fee8'


@unittest.skipUnless(SOURCE.is_dir(), 'Downloaded reference baseline is not available.')
class BaselineAuditTests(unittest.TestCase):
    def setUp(self):
        scratch_root = ROOT/'tmp'
        scratch_root.mkdir(exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix='baseline-audit-test-', dir=scratch_root)
        self.assertEqual(Path(self.scratch.name).resolve().parent, scratch_root.resolve())
        self.addCleanup(self.scratch.cleanup)
        self.folder = Path(self.scratch.name)/'records'
        shutil.copytree(SOURCE, self.folder)

    def change(self, name, transform):
        path = self.folder/name
        value = json.loads(path.read_text(encoding='utf-8'))
        transform(value)
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_downloaded_record_passes(self):
        result = audit(self.folder)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['evaluation_metrics']['itl_ms']['n'], 35)
        self.assertAlmostEqual(result['evaluation_metrics']['ttft_ms']['mean'], 9834.0621018)

    def test_modified_workload_rejected(self):
        self.change('workload.json', lambda value:value.update(new_tokens_cap=7))
        with self.assertRaisesRegex(ValueError, 'hash'):
            audit(self.folder)

    def test_modified_aggregate_rejected(self):
        self.change('performance.json', lambda value:value['evaluation_metrics']['ttft_ms'].update(mean=1))
        with self.assertRaisesRegex(ValueError, 'differs from raw'):
            audit(self.folder)

    def test_modified_raw_timestamp_rejected(self):
        path = self.folder/'request_metrics.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        rows[1]['token_completion_offsets_ms'][0] += 1
        path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'TTFT differs'):
            audit(self.folder)


if __name__ == '__main__':
    unittest.main()
