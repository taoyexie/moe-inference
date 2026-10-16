"""Token construction and request-ID tests only, not GPU execution."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from multi_trace import build_ids, request_plan
from kite_trace import ExpertRecorder


class MultiTraceTests(unittest.TestCase):
    def test_wrapper_preserved_and_exact_length(self):
        self.assertEqual(build_ids([1,2],[3,4,5],[6,7],9),[1,2,3,4,5,3,4,6,7])
        with self.assertRaises(ValueError):
            build_ids([1,2],[3],[4,5],4)

    def test_six_unique_requests(self):
        plan = request_plan([128,512,1024])
        self.assertEqual(len(plan),6)
        self.assertEqual(len({r['request_id'] for r in plan}),6)

    def test_recorder_identity_does_not_default_to_smoke(self):
        recorder = ExpertRecorder([0],8,request_id='trace-128-systems')
        recorder.begin(0,'prefill',128,0)
        self.assertEqual(recorder.context['request_id'],'trace-128-systems')
        default = ExpertRecorder([0],8)
        default.begin(0,'prefill',1,0)
        self.assertEqual(default.context['request_id'],'smoke-0001')


if __name__ == '__main__':
    unittest.main()
