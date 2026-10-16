"""Offline auditor unit tests with deliberately synthetic metadata, not experiment evidence."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_kv_paths import check_stats, tensor_header
from kv_paths import object_spec


class KVAuditTests(unittest.TestCase):
    def test_recomputed_stats_must_match(self):
        check_stats({'n':3,'median_ms':1.2},{'n':3,'median_ms':1.2})
        with self.assertRaises(ValueError):
            check_stats({'n':3,'median_ms':1.3},{'n':3,'median_ms':1.2})
        with self.assertRaises(ValueError):
            check_stats({'n':3},{'n':3,'median_ms':1.2})

    def test_payload_shape_and_length(self):
        spec = object_spec(1)
        header = json.dumps({'kv':{'dtype':'BF16','shape':spec['shape'],
                                  'data_offsets':[0,spec['bytes']]}}).encode()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path = Path(temporary)/'fixture.safetensors'
            path.write_bytes(struct.pack('<Q',len(header))+header+bytes(spec['bytes']))
            self.assertEqual(tensor_header(path,spec)['dtype'],'BF16')
            with self.assertRaises(ValueError):
                tensor_header(path,object_spec(2))
            path.write_bytes(struct.pack('<Q',len(header))+header)
            with self.assertRaises(ValueError):
                tensor_header(path,spec)


if __name__ == '__main__':
    unittest.main()
