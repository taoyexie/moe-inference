#!/usr/bin/env python3
"""Diagnose short mixed-batch correctness; does not run any performance workload."""
import argparse
import os
import time
import traceback
from pathlib import Path

from expert_paths import new_output
from kite_advantage import workload
from kite_e2e import ResourceMonitor
from kite_e2e_torch import load_engine
from kite_numerical_diagnostic import diagnose
from kite_readiness import readiness,require_ready
from kite_trace import GIB,write_json
from qwen_smoke import initialize_cuda_stats,show


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--results-dir',default='/workspace/results/kite_numerics')
    args = parser.parse_args()
    output = new_output(args.results_dir,'packed-diagnostic')
    state = dict(status='running',stage='readiness',is_performance=False,checkpoint=str(args.checkpoint))
    monitor = None
    started = time.perf_counter_ns()
    try:
        ready = readiness(args.checkpoint,Path(__file__).resolve().parents[1])
        write_json(output/'readiness.json',ready)
        require_ready(ready)
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.version.cuda!='11.8':
            raise RuntimeError('Expected validated one-GPU cu118 installation.')
        os.sched_setaffinity(0,set(range(24,32)))
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        device = torch.device('cuda:0')
        initialize_cuda_stats(torch,device)
        monitor = ResourceMonitor(evidence_path=output/'gpu_monitor_startup.json')
        with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            state['stage'] = 'model_loading'
            write_json(output/'summary.json',state)
            show('Loading once for short numerical diagnosis only. No main workload or calibration.')
            engine,tokenizer,state['loading'] = load_engine(torch,args.checkpoint,device,8*GIB,
                ready['resources'],health_check=monitor.check)
            engine.health_check = monitor.check
            engine.pad_attention_keys = True
            state['stage'] = 'diagnostic'
            write_json(output/'summary.json',state)
            report = diagnose(engine,workload(tokenizer,'search'),
                lambda data:write_json(output/'numerical_diagnostic.json',data))
            monitor.check()
            state.update(status='diagnostic_complete',stage='complete',
                within_original_tolerances=report['all_stock_reference_checks_within_original_tolerance'])
            show('Numerical diagnosis complete; no performance result was measured.')
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error))
        (output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        if monitor is not None:
            monitor.close()
            write_json(output/'resource_monitor.json',monitor.record())
            if monitor.error:
                state.update(status='failed',error_type='ResourceInterference',error=monitor.error)
        state['diagnostic_job_wall_ms_not_performance'] = (time.perf_counter_ns()-started)/1e6
        write_json(output/'summary.json',state)
        print('Results:',output,flush=True)
    if state['status']=='failed':
        raise RuntimeError(state.get('error','Diagnostic failed.'))


if __name__=='__main__':
    main()
