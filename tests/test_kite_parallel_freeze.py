"""Read-only evidence/freeze consistency; no GPU, tokenizer or model loading."""
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from kite_parallel_decode import implementation_hash,mode_order,protocol_hash
from kite_advantage import CONFIRM_TOPICS

FOLDER = ROOT/'reports'/'kite_parallel_decode_20261016'
MANIFEST = FOLDER/'confirmation'/'frozen_manifest.json'


@unittest.skipUnless(MANIFEST.is_file(),'Local frozen confirmation evidence unavailable.')
class ParallelFreezeTests(unittest.TestCase):
    def test_hashes_pin_successful_actual_search_and_audit(self):
        frozen = json.loads(MANIFEST.read_text(encoding='utf-8'))
        source = ROOT/'results'/'parallel-decode-search-20261016T132937Z-8668e263'
        targets = {'selection_sha256':source/'selection.json','search_summary_sha256':source/'summary.json',
            'search_audit_sha256':FOLDER/'search_seed17_audit'/'audit.json',
            'source_bundle_sha256':ROOT/'dist'/frozen['source_bundle']}
        for key,path in targets.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),frozen[key])
        selected = json.loads((source/'selection.json').read_text())
        summary = json.loads((source/'summary.json').read_text())
        self.assertEqual(summary['status'],'passed')
        self.assertEqual(summary['selection'],selected)
        self.assertEqual(selected['selected']['budget_mib'],frozen['budget_mib'])
        self.assertEqual(selected['selected']['case_name'],frozen['selected_case'])
        self.assertEqual(frozen['implementation_sha256'],implementation_hash())
        self.assertEqual(frozen['protocol_sha256'],protocol_hash())

    def test_confirmation_topics_and_declared_pair_order_are_pinned(self):
        frozen = json.loads(MANIFEST.read_text(encoding='utf-8'))
        actual = hashlib.sha256(json.dumps(CONFIRM_TOPICS,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        self.assertEqual(actual,frozen['held_out_topics_sha256'])
        self.assertEqual([mode_order(0,s) for s in frozen['confirmation_seeds']],frozen['mode_orders'])
        self.assertEqual(frozen['budget_mib'],6144)
        self.assertFalse(frozen['new_gpu_measurements_completed'])


if __name__=='__main__':
    unittest.main()
