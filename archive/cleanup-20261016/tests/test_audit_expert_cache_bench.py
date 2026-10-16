"""Downloaded-record audit regression checks; no GPU dependency."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_expert_cache_bench import audit

SOURCE = ROOT/'results'/'expert-cache-bench-20261016T012157Z-3d019646'


@unittest.skipUnless(SOURCE.is_dir(),'Downloaded benchmark is unavailable.')
class AuditTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='expert-bench-audit-')
        self.assertEqual(Path(self.temp.name).resolve().parent,(ROOT/'tmp').resolve())
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)/'records'
        shutil.copytree(SOURCE,self.folder)

    def change_passes(self, transform):
        path = self.folder/'passes.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        transform(rows)
        path.write_text('\n'.join(json.dumps(row) for row in rows),encoding='utf-8')

    def test_real_record_passes(self):
        result = audit(self.folder)
        self.assertEqual(result['formal_calls'],11520)
        self.assertEqual(result['detailed_events_checked'],1152)
        self.assertAlmostEqual(result['cases']['mixed_q_pressure']['paired_savings_mean'],.45571192545508143)

    def test_missing_pair_rejected(self):
        self.change_passes(lambda rows:rows.pop())
        with self.assertRaisesRegex(ValueError,'execution order/coverage'):
            audit(self.folder)

    def test_counter_change_rejected(self):
        self.change_passes(lambda rows:rows[0]['counters'].update(misses=1))
        with self.assertRaisesRegex(ValueError,'path counters'):
            audit(self.folder)

    def test_timing_summary_change_rejected(self):
        path = self.folder/'summary.json'
        summary = json.loads(path.read_text(encoding='utf-8'))
        summary['cases']['mixed_q_pressure']['paired_savings_mean'] = .9
        path.write_text(json.dumps(summary),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'timing/aggregate'):
            audit(self.folder)


if __name__=='__main__':
    unittest.main()
