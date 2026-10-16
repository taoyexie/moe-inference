"""Offline capacity sensitivity on recorded four-request decode accesses.

Reference LRU replay only: no CUDA, future routing prediction, or E2E estimate.
Empty reference caches start at each observed four-request decode segment.
"""
import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

from kite_controller import Controller
from kite_e2e_core import EXPERT_BYTES,FRAME_BYTES,KV_BYTES


def replay_segment(rows,expert_capacity,kv_capacity):
    caches = {'expert':OrderedDict(),'kv':OrderedDict()}
    misses = {'expert':0,'kv':0}
    accesses = {'expert':0,'kv':0}
    def access(kind,key,capacity,count=True):
        cache = caches[kind]
        if count:
            accesses[kind] += 1
        if key in cache:
            cache.move_to_end(key)
        else:
            if count:
                misses[kind] += 1
            cache[key] = None
            if len(cache)>capacity:
                cache.popitem(last=False)
    for row in rows:
        for group in row['kv_read_groups']:
            for key in group['objects']:
                access('kv',tuple(key),kv_capacity)
        for key,q in row['expert_accesses']:
            access('expert',tuple(key),expert_capacity)
        # Decode writes update precisely the page containing the consumed token.
        for group in row['kv_read_groups']:
            key = (group['group_id'][0],group['context']//128)
            access('kv',key,kv_capacity,count=False)
    return dict(accesses=accesses,misses=misses)


def screen(run,budgets=(6144,7168,8192,10240,12288)):
    run = Path(run)
    results = []
    for source in ('baseline','kite'):
        path = run/f'8192-{source}'/'batches.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        segments = []
        for row in rows:
            if row['prefill']==0 and row['decode']==4:
                if not segments or segments[-1][-1]['step']!=row['step']-1:
                    segments.append([])
                segments[-1].append(row)
        for budget in budgets:
            usable = budget*1024**2//FRAME_BYTES*FRAME_BYTES
            for history_fraction in (.5,.9):
                pair = {}
                for mode,weight in (('baseline',0.),('kite',.65)):
                    controller = Controller(usable,current_weight=weight)
                    controller.proposal = min(controller.choices,key=lambda c:abs(c.expert_bytes-usable*history_fraction))
                    controller.costs = object()  # Conditional split arithmetic, not an imposed runtime proposal.
                    candidate = controller.before_batch(0,4)['candidate']
                    samples = [replay_segment(segment,candidate['expert_objects'],candidate['kv_objects']) for segment in segments]
                    pair[mode] = dict(candidate=candidate,
                        expert_reference_misses=sum(s['misses']['expert'] for s in samples),
                        kv_reference_misses=sum(s['misses']['kv'] for s in samples))
                results.append(dict(source=source,budget_mib=budget,conditional_history_fraction=history_fraction,
                    segments=len(segments),decode_batches=sum(map(len,segments)),policies=pair,
                    kv_reference_misses_avoided=pair['baseline']['kv_reference_misses']-pair['kite']['kv_reference_misses'],
                    extra_expert_reference_misses=pair['kite']['expert_reference_misses']-pair['baseline']['expert_reference_misses']))
    return dict(scope='conditional admit-on-access LRU capacity replay of existing real accesses; NOT actual KITE misses, future trace, or E2E speedup',
        source_run=str(run.resolve()),assumptions=['fixed conditional history proposal within each segment',
            'empty reference caches at each segment start','CPU bypass not simulated',
            'do not transfer these numerical miss counts to different prompt/context inputs'],
        source_sha256={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [run/f'8192-{m}'/'batches.jsonl' for m in ('baseline','kite')]},rows=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    result = screen(args.run)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    for row in result['rows']:
        if row['conditional_history_fraction']==.5:
            print(row['source'],row['budget_mib'],'MiB; KV reference misses avoided:',
                row['kv_reference_misses_avoided'],'extra Expert reference misses:',row['extra_expert_reference_misses'])


if __name__=='__main__':
    main()
