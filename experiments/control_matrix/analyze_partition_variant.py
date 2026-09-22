#!/usr/bin/env python3
"""Audit a matched partition-method experiment without refitting its threshold."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import sys
from pathlib import Path
import subprocess

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--tasks', default='tworoom,pusht,reacher,cube')
    p.add_argument('--num-clusters', type=int, default=4)
    p.add_argument('--method', default='kmeanspp')
    p.add_argument('--partition-seeds', default='0,1,2')
    p.add_argument('--training-seeds', default='0,42,625')
    p.add_argument('--evaluation-seeds', default='0,1,2,3,4')
    p.add_argument('--ridge', type=float, default=1e-8)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--phase', choices=('preflight', 'score', 'finalize'), required=True)
    p.add_argument('--threshold-policy', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    repo = a.repo.resolve()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    seeds = lambda s: [int(v) for v in s.split(',')]
    ps, ts, es = map(seeds, (a.partition_seeds, a.training_seeds, a.evaluation_seeds))
    policy = a.threshold_policy.resolve()
    threshold = float(read(policy)['frozen_bures_threshold'])
    tasks = a.tasks.split(',')
    roots = {t: repo / f'experiments/{t}/matrix_k{a.num_clusters}' for t in tasks}
    longs = {t: repo / f'experiments/{t}/matrix_k{a.num_clusters}_long' for t in tasks}
    common = dict(repository_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                  model='LeWM', method=a.method, num_clusters=a.num_clusters,
                  partition_seeds=ps, training_seeds=ts, evaluation_seeds=es,
                  goal_offset_steps=50, eval_budget=50, num_eval=50,
                  threshold=threshold, threshold_policy=str(policy), threshold_policy_sha256=digest(policy),
                  threshold_refitted=False, rule='Regional iff mean weakest-pair Bures > frozen threshold; no Check 1',
                  aggregation='weakest pair within each partition seed, then arithmetic mean across partition seeds',
                  scope='partition-method transfer at development K; excluded from original Layers 2+3 totals')
    common['environment'] = dict(host=platform.node(), python=sys.version, numpy=np.__version__)
    common['implementation_sha256'] = {name: digest(repo / 'experiments/control_matrix' / name) for name in (
        'analyze_partition_variant.py', 'analyze_fixed_k_response_geometry.py', 'fit_partition.py',
        'train_predictors.py', 'scripts/run_lewm_partition_variant.sh', 'scripts/run_lewm_k2.sh',
        'scripts/run_lewm_matrix.sh', 'scripts/run_lewm_matrix_parallel.sh', 'scripts/wait_for_free_gpu.sh')}
    common['routing_implementation_sha256'] = {name: digest(repo / name) for name in (
        'lap/routing/voronoi.py', 'lap/routing/gaussian_mixture.py',
        'experiments/tworoom/tworoom_success_rate_eval.py')}
    if a.phase == 'preflight':
        for t in tasks:
            assert (roots[t] / 'preparation/embedding_cache.npz').is_file(), t
            assert (longs[t] / 'training').resolve() == (roots[t] / 'training').resolve(), t
            for e in es:
                official = read(longs[t] / f'eval/official/eval{e}/results.json')
                assert len(official['eval_start_indices']) == 50, (t, e)
                assert official['goal_offset_steps'] == official['eval_budget'] == 50, (t, e)
                for tr in ts:
                    g = read(longs[t] / f'eval/global/train{tr}/eval{e}/results.json')
                    assert g['eval_start_indices'] == official['eval_start_indices'], (t, tr, e)
                    assert g['goal_offset_steps'] == g['eval_budget'] == 50
                for part, tr in itertools.product(ps, ts):
                    s = read(longs[t] / f'eval/spectral/partition{part}_train{tr}/eval{e}/results.json')
                    assert s['eval_start_indices'] == official['eval_start_indices'], (t, part, tr, e)
                    assert s['goal_offset_steps'] == s['eval_budget'] == 50
            print(f'[preflight] {t}: source cache, training symlink, 5 official, 15 Global and 45 Spectral paired results verified', flush=True)
        (out / 'preflight.json').write_text(json.dumps(common, indent=2) + '\n')
        return
    if a.phase == 'score':
        os.environ['OMP_NUM_THREADS'] = str(a.cpu_threads)
        import analyze_fixed_k_response_geometry as rg
        seed_rows, pair_rows = [], []
        for t in tasks:
            manifests = {s: read(roots[t] / f'partitions/{a.method}/seed{s}/manifest.json') for s in ps}
            assert all(m['num_clusters'] == a.num_clusters and m['method'] == a.method for m in manifests.values())
            source = read(roots[t] / 'partitions/spectral/seed0/manifest.json')
            assert all(m['partition_seed'] == s and m['latent_cache_sha256'] == source['latent_cache_sha256'] for s, m in manifests.items())
            x, ids = rg.load_unique(roots[t] / 'preparation/embedding_cache.npz', 5)
            labels = {s: rg.load_labels(roots[t] / f'partitions/{a.method}/seed{s}/cluster_labels.npz', ids) for s in ps}
            assert all(set(np.unique(y)) == set(range(a.num_clusters)) for y in labels.values())
            left, right, acts = rg.transition_rows(ids, Path(manifests[ps[0]]['data_file']), 1)
            fitted, _, cov, _, _ = rg.sufficient_stats(x, left, right, acts, labels, a.num_clusters, a.ridge, 100000)
            for s in ps:
                vals = []
                for i, j in itertools.combinations(range(a.num_clusters), 2):
                    score = rg.jacobian_metrics(fitted[s][i][0], fitted[s][j][0], cov)[3]
                    vals.append(score)
                    pair_rows.append(dict(task=t, method=a.method, num_clusters=a.num_clusters, partition_seed=s, region_i=i, region_j=j, normalized_squared_bures=score))
                seed_rows.append(dict(task=t, method=a.method, num_clusters=a.num_clusters, partition_seed=s, weakest_pair_bures=min(vals), transition_count=len(left)))
            print(f'[score] {t} complete; transition count={len(left)}', flush=True)
            del x, ids, left, right, acts, labels, fitted
        write_csv(out / 'bures_by_seed.csv', seed_rows)
        write_csv(out / 'bures_pair_metrics.csv', pair_rows)
        common['ridge'] = a.ridge
        common['transition_stride'] = 1
        common['score_files_sha256'] = {n: digest(out / n) for n in ('bures_by_seed.csv', 'bures_pair_metrics.csv')}
        (out / 'score_manifest.json').write_text(json.dumps(common, indent=2) + '\n')
        return
    with (out / 'bures_by_seed.csv').open() as f:
        scores = list(csv.DictReader(f))
    rows, raw = [], []
    for t in tasks:
        regional, global_values = [], []
        for part, tr, e in itertools.product(ps, ts, es):
            path = longs[t] / f'eval/{a.method}/partition{part}_train{tr}/eval{e}/results.json'
            r = read(path)
            gpath = longs[t] / f'eval/global/train{tr}/eval{e}/results.json'
            g = read(gpath)
            ref = read(longs[t] / f'eval/official/eval{e}/results.json')
            assert r['eval_start_indices'] == g['eval_start_indices'] == ref['eval_start_indices'], str(path)
            assert r['goal_offset_steps'] == r['eval_budget'] == r['num_eval'] == 50
            assert r['num_clusters'] == a.num_clusters and r['latent_routing'] == 'mpc'
            manifest = read(roots[t] / f'training/{a.method}/partition{part}_train{tr}/manifest.json')
            baseline = read(roots[t] / f'training/spectral/partition{part}_train{tr}/manifest.json')
            assert manifest['training_config'] == baseline['training_config'], str(path)
            assert manifest['pretrained_model_sha256'] == baseline['pretrained_model_sha256']
            assert manifest['latent_cache_sha256'] == baseline['latent_cache_sha256']
            rv, gv = float(r['metrics']['success_rate']), float(g['metrics']['success_rate'])
            regional.append(rv)
            raw.append(dict(task=t, partition_seed=part, train_seed=tr, eval_seed=e, regional_percent=rv, global_percent=gv, delta_pp=rv-gv,
                            regional_source=str(path.relative_to(repo)), regional_sha256=digest(path), global_source=str(gpath.relative_to(repo)), global_sha256=digest(gpath)))
        for tr, e in itertools.product(ts, es):
            global_values.append(float(read(longs[t] / f'eval/global/train{tr}/eval{e}/results.json')['metrics']['success_rate']))
        mean_r, mean_g = float(np.mean(regional)), float(np.mean(global_values))
        values = [float(v['weakest_pair_bures']) for v in scores if v['task'] == t]
        assert len(values) == len(ps)
        b = float(np.mean(values))
        predicted = 'Regional' if b > threshold else 'Global'
        winner = 'Regional' if mean_r > mean_g else 'Global' if mean_r < mean_g else 'Tie'
        rtrain = [np.mean([z['regional_percent'] for z in raw if z['task'] == t and z['train_seed'] == tr]) for tr in ts]
        gtrain = [np.mean([z['global_percent'] for z in raw if z['task'] == t and z['train_seed'] == tr]) for tr in ts]
        rows.append(dict(task=t, model='LeWM', method=a.method, num_clusters=a.num_clusters, bures_mean=b, frozen_threshold=threshold,
                         regional_mean_percent=mean_r, global_mean_percent=mean_g, delta_regional_minus_global_pp=mean_r-mean_g,
                         regional_sd_across_train_seeds=float(np.std(rtrain, ddof=1)), global_sd_across_train_seeds=float(np.std(gtrain, ddof=1)),
                         predicted_branch=predicted, point_estimate_winner=winner, selection_correct=predicted==winner,
                         near_zero_0p4pp=abs(mean_r-mean_g)<=0.4, paired_starts_verified=True))
    write_csv(out / 'long_comparison.csv', rows)
    write_csv(out / 'long_raw.csv', raw)
    common['accuracy'] = dict(correct=sum(r['selection_correct'] for r in rows), total=len(rows), ties=sum(r['point_estimate_winner']=='Tie' for r in rows))
    common['files_sha256'] = {n: digest(out/n) for n in ('long_comparison.csv', 'long_raw.csv', 'bures_by_seed.csv', 'bures_pair_metrics.csv')}
    common['rows'] = rows
    (out / 'experiment_manifest.json').write_text(json.dumps(common, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
