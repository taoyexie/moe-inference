import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from expert_cache import ExpertCache, ExpertCosts, fit_costs


class FakeBackend:
    def __init__(self):
        self.next_slot, self.live, self.events = 0, {}, []
        self.fail_upload = False

    def synchronize(self):
        self.events.append('sync')

    def allocate_slot(self):
        self.next_slot += 1
        self.live[self.next_slot] = None
        return self.next_slot

    def release_slot(self, slot):
        del self.live[slot]

    def upload(self, key, slot):
        self.live[slot] = key
        if self.fail_upload:
            raise RuntimeError('simulated upload failure')

    def gpu(self, slot, x):
        return self.live[slot], x

    def cpu(self, key, x):
        return key, x

    def scratch_upload(self, key, x):
        return key, x


class ExpertCacheTests(unittest.TestCase):
    def make(self, capacity=2, policy='upload', costs=None):
        backend = FakeBackend()
        cache = ExpertCache(backend, 9, capacity, policy, costs)
        return backend, cache

    def test_lru_admission_and_eviction(self):
        backend, cache = self.make()
        records = [cache.execute(key, 1, 'activation')[1] for key in 'ABACBAA']
        self.assertEqual([r['hit'] for r in records], [False,False,True,False,False,False,True])
        self.assertEqual([r['evicted'] for r in records if r['evicted']], list('BAC'))
        self.assertEqual(cache.counts['misses'], 5)
        self.assertEqual(cache.counts['hits'], 2)
        self.assertEqual(len(backend.live), 2)
        self.assertEqual(cache.allocated_weight_bytes, 18)

    def test_cpu_bypass_does_not_promote_or_insert(self):
        costs = ExpertCosts(1, .1, .1, .5, 1, 20)
        _, cache = self.make(policy='calibrated', costs=costs)
        cache.execute('A', 20, None)
        cache.execute('B', 20, None)
        _, row = cache.execute('C', 1, None)
        self.assertEqual(row['path'], 'cpu')
        self.assertFalse(row['admitted'])
        self.assertEqual(list(cache.resident), ['A','B'])
        _, hit = cache.execute('A', 1, None)
        self.assertEqual(hit['path'], 'hit')
        self.assertEqual(list(cache.resident), ['B','A'])

    def test_resize_preserves_mru_and_releases_actual_slots(self):
        backend, cache = self.make()
        for key in 'ABA':
            cache.execute(key, 1, None)
        self.assertEqual(cache.resize(1)['evicted'], ['B'])
        self.assertEqual(len(backend.live), 1)
        self.assertEqual(list(cache.resident), ['A'])
        cache.resize(3)
        self.assertEqual(len(backend.live), 3)
        cache.close()
        cache.close()
        self.assertFalse(backend.live)
        with self.assertRaises(RuntimeError):
            cache.execute('A', 1, None)

    def test_zero_capacity_uses_scratch_without_admission(self):
        backend, cache = self.make(capacity=0)
        for _ in range(2):
            output, row = cache.execute('A', 1, 'x')
            self.assertEqual(output, ('A','x'))
            self.assertFalse(row['hit'] or row['admitted'])
        self.assertFalse(backend.live)
        self.assertEqual(cache.allocated_weight_bytes, 0)

    def test_upload_failure_does_not_keep_stale_key(self):
        backend, cache = self.make(capacity=1)
        cache.execute('A', 1, None)
        backend.fail_upload = True
        with self.assertRaises(RuntimeError):
            cache.execute('B', 1, None)
        self.assertNotIn('A', cache.resident)
        self.assertNotIn('B', cache.resident)
        self.assertEqual(len(cache.free), 1)
        self.assertEqual(cache.counts['calls'], 1)

    def test_measured_fit_and_domain_guard(self):
        samples = [dict(path=path, q=q, wall_ms=(1 if path=='upload' else .1+.1*q))
                   for q in (1,4,16) for path in ('upload','cpu')]
        costs = fit_costs(samples)
        self.assertEqual(costs.choose(1)[0], 'cpu')
        self.assertEqual(costs.choose(16)[0], 'upload')
        self.assertEqual(costs.choose(32), ('upload','outside_calibration_domain_upload'))
        self.assertAlmostEqual(costs.cpu_slope_ms, .1)
        with self.assertRaises(ValueError):
            fit_costs(samples[:-1])

    def test_invalid_negative_cpu_prediction_fails_closed_to_upload(self):
        costs = ExpertCosts(1, -10, 0, 1, 1, 20)
        self.assertEqual(costs.choose(2), ('upload','invalid_cpu_prediction_upload'))

    def test_empty_or_invalid_capacities_rejected(self):
        for capacity in (-1, .5, True):
            with self.assertRaises(ValueError):
                self.make(capacity=capacity)
        _, cache = self.make()
        with self.assertRaises(ValueError):
            cache.execute('A', 0, None)


if __name__ == '__main__':
    unittest.main()
