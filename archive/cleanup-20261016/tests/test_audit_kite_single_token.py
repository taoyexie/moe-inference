"""Offline audit bookkeeping for one-token requests; no model/GPU execution."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_kite_e2e as audit_module
from kite_burst_search import OPTIONS

SOURCE = ROOT/'results'/'prefill-burst-search-20261016T124129Z-e0088d2f'


@unittest.skipUnless(SOURCE.is_dir(),'Downloaded single-token workload is unavailable.')
class SingleTokenAuditTests(unittest.TestCase):
    def setUp(self):
        self.folder = SOURCE/'repeat-20-32768-baseline'
        self.specs = audit_module.read(SOURCE/'workload.json')['cases']['repeat-20']

    def test_single_output_has_no_tpot(self):
        result = audit_module.audit_task(self.folder,self.specs,'baseline',32768,OPTIONS)
        self.assertEqual(result['output_tokens'],16)
        rows = audit_module.read(self.folder/'requests.json')
        self.assertTrue(all(r['tpot_ms'] is None and r['itl_ms']==[] for r in rows))

    def test_zero_tpot_is_not_interchangeable_with_null(self):
        original = audit_module.read
        def altered(path):
            result = original(path)
            if Path(path).name=='requests.json':
                result[0]['tpot_ms'] = 0
            return result
        with patch.object(audit_module,'read',side_effect=altered):
            with self.assertRaisesRegex(ValueError,'Expected null statistic'):
                audit_module.audit_task(self.folder,self.specs,'baseline',32768,OPTIONS)


if __name__=='__main__':
    unittest.main()
