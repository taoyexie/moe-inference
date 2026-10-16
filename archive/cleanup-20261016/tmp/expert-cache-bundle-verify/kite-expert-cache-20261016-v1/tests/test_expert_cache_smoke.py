"""Acceptance-driver tests with fake math; real CUDA validation remains remote."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from expert_cache import ExpertCosts
from expert_cache_smoke import read_capture, run_scenarios, schedule
from test_expert_cache import FakeBackend


class DriverBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.sources = {(0,15):{},(23,96):{},(47,56):{}}
        self.device = 'fake'
        self.torch = SimpleNamespace(cuda=SimpleNamespace(memory_allocated=lambda device:0))

    def activation(self, key, q):
        return SimpleNamespace(shape=(q or 6,2048)), {'native_q':6,'input_source':'fake_test_only'}

    def validate(self, key, x, value):
        if value != (key,x):
            raise ValueError('Fake backend changed expert identity.')
        return {'relative_l2':0,'normalized_max_abs':0,'max_abs':0}


class DriverTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='expert-cache-test-')
        self.assertEqual(Path(self.scratch.name).resolve().parent, (ROOT/'tmp').resolve())
        self.addCleanup(self.scratch.cleanup)
        self.folder = Path(self.scratch.name)

    def test_schedule_is_explicitly_controlled(self):
        self.assertEqual(len(schedule(['A','B','C'])), 12)
        self.assertEqual(schedule(['A','B','C'])[9], ('B',32))
        with self.assertRaises(ValueError):
            schedule(['A','B','B'])

    def test_five_scenarios_reconcile_and_release_slots(self):
        backend = DriverBackend()
        result = run_scenarios(backend,ExpertCosts(1,.1,.1,1.2,1,32),self.folder,2)
        rows = [json.loads(line) for line in (self.folder/'events.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),60)
        self.assertEqual(len(result),5)
        self.assertFalse(backend.live)
        for scenario in result:
            events = [row for row in rows if row['scenario'] == scenario['scenario']]
            self.assertEqual(scenario['counters']['calls'],12)
            self.assertEqual(scenario['counters'].get('hits',0),sum(row['hit'] for row in events))
            self.assertTrue(all(row['allocated_weight_bytes'] <= row['capacity']*9*1024**2 for row in events))
        self.assertEqual(result[1]['counters']['misses'],12)
        self.assertTrue(all(not row['admitted'] for row in rows if row['scenario']=='cpu_bypass'))
        zero = [row for row in rows if row['scenario']=='resize_acceptance' and row['position']==6][0]
        self.assertEqual(zero['allocated_weight_bytes'],0)
        self.assertFalse(zero['admitted'] or zero['hit'])

    def capture(self):
        manifest = {'content_manifest_sha256':'identity','source':'modelscope','revision':'master'}
        samples = []
        for layer, expert in ((0,15),(23,96),(47,56)):
            name = f'{layer}.safetensors'
            (self.folder/name).write_bytes(b'not-real-activation-test-only')
            samples.append(dict(layer=layer,expert=expert,file=name,
                                sha256=hashlib.sha256((self.folder/name).read_bytes()).hexdigest()))
        capture = dict(status='passed',model_content_manifest_sha256='identity',
                       model_source='modelscope',model_revision='master',samples=samples)
        (self.folder/'capture.json').write_text(json.dumps(capture),encoding='utf-8')
        return capture,manifest

    def test_capture_identity_and_hash_gate(self):
        _, manifest = self.capture()
        self.assertEqual(len(read_capture(self.folder,manifest)['samples']),3)
        (self.folder/'0.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'hash differs'):
            read_capture(self.folder,manifest)

    def test_capture_duplicate_rejected(self):
        captured, manifest = self.capture()
        captured['samples'][1] = captured['samples'][0]
        (self.folder/'capture.json').write_text(json.dumps(captured),encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):
            read_capture(self.folder,manifest)


if __name__ == '__main__':
    unittest.main()
