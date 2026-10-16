"""NumPy-backed tensor adapter tests, NOT validation of CUDA or Qwen kernels."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
try:
    import numpy as np
except ImportError:
    local_deps = ROOT/'tmp'/'plot-deps'
    if local_deps.is_dir():
        sys.path.append(str(local_deps))
    try:
        import numpy as np
    except ImportError:
        np = None

from kite_e2e_core import FramePool,Request,packed_layout
from kite_e2e_torch import QwenEngine,TorchArena,batch_tensors,make_packed_cache,packed_forward
from kite_e2e_runtime import Runtime


class Tensor:
    def __init__(self,data):
        self.data = data

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self,index):
        return Tensor(self.data[index])

    def copy_(self,source,**kwargs):
        np.copyto(self.data,source.data)
        return self

    def view(self,*shape):
        return Tensor(self.data.reshape(shape))

    def unsqueeze(self,axis):
        return Tensor(np.expand_dims(self.data,axis))

    def masked_fill_(self,mask,value):
        self.data[mask.data] = value
        return self

    def __le__(self,other):
        return Tensor(self.data<=other.data)

    def numel(self):
        return self.data.size

    def element_size(self):
        return 2 if self.data.dtype==np.float32 else self.data.itemsize

    def zero_(self):
        self.data.fill(0)
        return self

    def clone(self):
        return Tensor(self.data.copy())


class Torch:
    @property
    def bfloat16(self):
        return np.float32  # Shapes/copy logic only; no BF16 approximation claim.

    @property
    def long(self):
        return np.int64

    def empty(self,shape,dtype=None,**kwargs):
        return Tensor(np.empty(shape,dtype=dtype))

    def zeros(self,shape,dtype=None,**kwargs):
        return Tensor(np.zeros(shape,dtype=dtype))

    def full(self,shape,value,dtype=None,**kwargs):
        return Tensor(np.full(shape,value,dtype=dtype))

    def tensor(self,data,dtype=None,**kwargs):
        return Tensor(np.array(data,dtype=dtype))

    def arange(self,*args,**kwargs):
        return Tensor(np.arange(*args))

    def finfo(self,dtype):
        return np.finfo(dtype)

    def cat(self,tensors,dim):
        return Tensor(np.concatenate([t.data for t in tensors],axis=dim))


@unittest.skipUnless(np is not None,'NumPy needed for tensor-copy adapter tests.')
class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.torch = Torch()
        self.d = Request('D',[1,2,3],0,2,generated=[4],kv_length=3)
        self.p = Request('P',[5,6],0,2)
        self.layout = packed_layout([self.d,self.p])

    def test_actual_mask_builder_has_no_cross_request_attention(self):
        tensors = batch_tensors(self.torch,[self.d,self.p],self.layout,'fake')
        np.testing.assert_array_equal(tensors['input_ids'].data,[[4,5,6]])
        np.testing.assert_array_equal(tensors['position_ids'].data,[[3,0,1]])
        np.testing.assert_array_equal(tensors['logits_to_keep'].data,[0,2])
        np.testing.assert_array_equal(tensors['attention_mask'].data[0,0]==0,
            [[1,1,1,1,0,0],[0,0,0,0,1,0],[0,0,0,0,1,1]])

    def test_cache_update_packs_different_contexts_without_mixing(self):
        dense = {'D':self.torch.zeros((48,2,1,4,4,128)),
                 'P':self.torch.zeros((48,2,1,4,2,128))}
        dense['D'].data[0,0,:,:,:3,:] = 7
        dense['D'].data[0,1,:,:,:3,:] = 9
        incoming = np.zeros((1,4,3,128))
        incoming[:,:,0,:],incoming[:,:,1,:],incoming[:,:,2,:] = 100,200,300
        with patch.dict(sys.modules,{'transformers':SimpleNamespace(Cache=object)}):
            cache = make_packed_cache(self.torch,self.layout,dense)
        key,value = cache.update(Tensor(incoming),Tensor(incoming+1),0)
        self.assertEqual(key.shape,(1,4,6,128))
        np.testing.assert_array_equal(key.data[0,0,:,0],[7,7,7,100,200,300])
        np.testing.assert_array_equal(value.data[0,0,:,0],[9,9,9,101,201,301])
        with self.assertRaises(RuntimeError):
            cache.update(Tensor(incoming),Tensor(incoming),0)

    def test_replay_is_fresh_full_prefix_with_the_normal_explicit_attention_adapter(self):
        engine = QwenEngine.__new__(QwenEngine)
        engine.torch,engine.device = self.torch,'fake'
        engine.in_replay,engine.replay_workspace_peak = False,0
        engine.events = ['existing normal observations']
        calls = []
        def model(**kwargs):
            self.assertTrue(engine.in_replay)
            calls.append(kwargs)
            ids = kwargs['input_ids'].data[0]
            n = len(ids)
            np.testing.assert_array_equal(kwargs['position_ids'].data,[list(range(n))])
            np.testing.assert_array_equal(kwargs['cache_position'].data,np.arange(n))
            np.testing.assert_array_equal(kwargs['logits_to_keep'].data,[n-1])
            np.testing.assert_array_equal(kwargs['attention_mask'].data[0,0]==0,
                                          np.tri(n,dtype=bool))
            cache = kwargs['past_key_values']
            self.assertEqual(cache.get_seq_length(),0)
            self.assertFalse(cache.updated)
            for layer in range(48):
                key = Tensor(np.broadcast_to(ids[None,None,:,None]+layer,(1,4,n,128)).copy())
                value = Tensor(key.data+1000)
                cache.update(key,value,layer)
            return SimpleNamespace(past_key_values=cache)
        engine.model = model
        with patch.dict(sys.modules,{'transformers':SimpleNamespace(Cache=object)}):
            recovered = engine.replay_prefix([11,22,33])
            second = engine.replay_prefix([99])
        self.assertEqual(len(calls),2)  # One full forward per prefix, not one per missing block.
        self.assertFalse(engine.in_replay)
        self.assertEqual(engine.events,['existing normal observations'])
        self.assertEqual(recovered.shape,(48,2,1,4,3,128))
        np.testing.assert_array_equal(recovered.data[47,0,0,0,:,0],[58,69,80])
        np.testing.assert_array_equal(recovered.data[47,1,0,0,:,0],[1058,1069,1080])
        self.assertEqual(second.data[0,0,0,0,0,0],99)
        self.assertFalse(np.shares_memory(recovered.data,second.data))
        self.assertIsNot(calls[0]['past_key_values'],calls[1]['past_key_values'])
        self.assertEqual(engine.replay_workspace_peak,3*48*2*4*128*2)

    def test_replay_exception_restores_tag_and_rejects_incomplete_kv(self):
        engine = QwenEngine.__new__(QwenEngine)
        engine.torch,engine.device = self.torch,'fake'
        engine.in_replay,engine.replay_workspace_peak = True,0
        engine.model = lambda **kwargs:SimpleNamespace()
        with patch.dict(sys.modules,{'transformers':SimpleNamespace(Cache=object)}):
            with self.assertRaisesRegex(RuntimeError,'all 48 KV layers'):
                engine.replay_prefix([1,2])
        self.assertTrue(engine.in_replay)  # Restore previous state, not unconditionally False.
        with self.assertRaisesRegex(RuntimeError,'workspace bound'):
            engine.replay_prefix([])

    def test_shared_forward_gate_rejects_partial_cache(self):
        dense = {'D':self.torch.zeros((48,2,1,4,4,128)),
                 'P':self.torch.zeros((48,2,1,4,2,128))}
        with patch.dict(sys.modules,{'transformers':SimpleNamespace(Cache=object)}):
            with self.assertRaisesRegex(RuntimeError,'all 48 KV layers'):
                packed_forward(self.torch,lambda **kwargs:None,[self.d,self.p],
                               self.layout,dense,'fake')

    def engine(self):
        torch = self.torch
        engine = SimpleNamespace(torch=torch,device='fake',synchronize=lambda:None,
                                 costs=None)
        def attach(budget):
            engine.pool = FramePool(budget)
            engine.arena = TorchArena(torch,engine.pool,'fake')
        engine.attach_pool = attach
        engine.reset_task = lambda:None
        engine.kv_scratch = torch.zeros((48,2,1,4,128,128),dtype=np.float32)
        return engine

    def initial_kv(self):
        engine = self.engine()
        runtime = Runtime(engine,21*1024**2,'baseline')
        request = Request('request',list(range(129)),0,3)
        dense = self.torch.empty((48,2,1,4,129,128),dtype=np.float32)
        dense.data[:] = np.arange(129)[None,None,None,None,:,None]
        runtime.commit(request,dense,0,129)
        request.generated.append(130)
        return engine,runtime,request,dense

    def test_kv_eviction_restore_tail_and_append_version(self):
        engine,runtime,request,reference = self.initial_kv()
        self.assertEqual(len(engine.pool.entries['kv']),1)
        dense,groups = runtime.prepare([request],packed_layout([request]))
        np.testing.assert_array_equal(dense['request'].data[:,:,:,:,:129,:],reference.data)
        self.assertEqual(runtime.counters['kv_misses'],2)  # Restoring first evicts the second.
        self.assertEqual(len(groups[0]['objects']),2)
        dense['request'].data[:,:,:,:,129,:] = 777
        runtime.commit(request,dense['request'],129,130)
        self.assertEqual(request.versions[1],2)
        self.assertTrue(np.all(request.cpu_pages[1].data[:,:,:,:,1,:]==777))
        self.assertTrue(np.all(request.cpu_pages[1].data[:,:,:,:,0,:]==128))
        entry = runtime.resident(request,1)
        entry['version'] = -1
        self.assertIsNone(runtime.resident(request,1))
        runtime.expire(request)
        self.assertEqual(runtime.host_kv_live,0)
        self.assertFalse(request.cpu_pages)

    def test_shared_replay_once_for_multiple_missing_pages(self):
        engine,runtime,request,reference = self.initial_kv()
        engine.pool.clear()
        calls = []
        def replay(tokens):
            calls.append(tokens)
            return reference.clone()
        engine.replay_prefix = replay
        dense,_ = runtime.prepare([request],packed_layout([request]),force_kv='replay')
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0],list(range(129)))
        self.assertEqual(runtime.counters['kv_misses'],2)
        self.assertEqual(runtime.counters['kv_replay_tasks'],1)
        self.assertEqual(len(runtime.recoveries),1)
        np.testing.assert_array_equal(dense['request'].data[:,:,:,:,:129,:],reference.data)


if __name__=='__main__':
    unittest.main()
