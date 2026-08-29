#!/usr/bin/env python3
"""Evaluate 22 frozen K=4 criteria on LeWM K=2,3 without threshold refitting."""
import argparse, hashlib, json, subprocess
from itertools import combinations
from pathlib import Path

import faiss, h5py, numpy as np, pandas as pd
from scipy.stats import spearmanr
from experiments.control_matrix import analyze_fixed_k_response_geometry as rg
from lap.partition.landmark import _sample_landmarks

TASKS=("tworoom","pusht","reacher","cube"); SEEDS=(0,1,2)
METRICS=("normalized_cluster_entropy","eigengap_after_k","prototype_distance_ratio",
"knn_purity","margin_radius_ratio_mean","tangent_contrast_mean_d8","curvature_tail_mean_d8",
"flow_persistence_h10","latent_velocity_eta2","action_residual_velocity_eta2",
"affine_response_contrast_ratio","minimum_pairwise_response","pairwise_uniformity_min_over_mean",
"response_centroid_spearman","response_boundary_spearman","boundary_min_jacobian_bures_distance",
"jacobian_bures_distance","jacobian_cosine_distance","jacobian_log_scale_distance",
"jacobian_subspace_chordal_distance","check1_retained_safety_fraction","check2_prominence_ratio")

def args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--repo",type=Path,required=True)
 p.add_argument("--model",choices=("lewm","subjepa"),default="lewm")
 p.add_argument("--tasks",default=",".join(TASKS)); p.add_argument("--clusters",default="2,3")
 p.add_argument("--partition-seeds",default="0,1,2"); p.add_argument("--frameskip",type=int,default=5)
 p.add_argument("--ridge",type=float,default=1e-8); p.add_argument("--chunk",type=int,default=100000)
 p.add_argument("--cpu-threads",type=int,default=8); p.add_argument("--audit-dir",type=Path,default=Path("/tmp/lap_k4_geometry_audit"))
 p.add_argument("--output-dir",type=Path,required=True); return p.parse_args()
def ints(s): return tuple(map(int,filter(None,s.split(","))))
def gate(repo,model,t,k):
 if model=="subjepa":
  if k==3: return repo/f"experiments/{t}/subjepa/formal/gate/partition/manifest.json"
  return root(repo,model,t,k,0)/"manifest.json"
 return repo/f"experiments/{t}/results/auto_gate_complete_k{k}/auto/partition/manifest.json"
def root(repo,model,t,k,s):
 if model=="subjepa":
  suffix="matrix" if k==3 else f"matrix_k{k}"
  return repo/f"experiments/{t}/subjepa/{suffix}/partitions/spectral/seed{s}"
 if k in (2,4): return repo/f"experiments/{t}/matrix_k{k}/partitions/spectral/seed{s}"
 if t=="tworoom": return repo/f"experiments/tworoom/results/latent_landmark_spectral_k3/spectral_M20000_k30_P16_seed{s}"
 return repo/f"experiments/{t}/matrix/partitions/spectral/seed{s}"
def protos(r):
 for b in (r/"partition",r):
  if (b/"routing_prototypes.npy").exists(): return b/"routing_prototypes.npy",b/"prototype_cluster_ids.npy"
 raise FileNotFoundError(r)
def meta(r):
 if (r/"manifest.json").exists():
  d=json.loads((r/"manifest.json").read_text())
  if "cluster_fractions" in d: return np.asarray(d["cluster_fractions"]),float(d["method_metadata"]["spectral"]["eigengap_after_k"])
 p=r/"partition/cluster_meta.json" if (r/"partition/cluster_meta.json").exists() else r/"cluster_meta.json"; d=json.loads(p.read_text()); f=d.get("full_cluster_fractions",d.get("cluster_fractions")); return (np.asarray(f) if f is not None else None),float(d["spectral"]["eigengap_after_k"])
def proj(u,v,d=8): return float(np.sqrt(max(0,1-np.linalg.norm(u[:,:d].T@v[:,:d],"fro")**2/d)))

def audit_metrics(x,ys,k):
 idx=faiss.IndexFlatIP(x.shape[1]); idx.add(np.ascontiguousarray(x)); _,nn=idx.search(np.ascontiguousarray(x),31); nn=nn[:,1:]
 out={}
 for seed,y in ys.items():
  rr={r:np.flatnonzero(y==r) for r in range(k)}; search={}
  same=np.empty(len(x),np.float32); other=np.full(len(x),np.inf,np.float32)
  for r in range(k):
   z=faiss.IndexFlatIP(x.shape[1]); z.add(np.ascontiguousarray(x[rr[r]])); search[r]=z
   sim,_=z.search(np.ascontiguousarray(x[rr[r]]),31); same[rr[r]]=1-sim[:,30]
  for r in range(k):
   for q in range(k):
    if q==r: continue
    sim,_=search[q].search(np.ascontiguousarray(x[rr[r]]),1); other[rr[r]]=np.minimum(other[rr[r]],1-sim[:,0])
  rng=np.random.default_rng(20260824); queries=np.concatenate([rng.choice(rr[r],128,replace=False) for r in range(k)])
  within={}; cross={}
  for row in queries:
   r=int(y[row]); _,p=search[r].search(np.ascontiguousarray(x[row:row+1]),2); within[int(row)]=int(rr[r][p[0,1]])
   best=(-np.inf,None)
   for q in range(k):
    if q==r: continue
    sim,p=search[q].search(np.ascontiguousarray(x[row:row+1]),1)
    if sim[0,0]>best[0]: best=(float(sim[0,0]),int(rr[q][p[0,0]]))
   cross[int(row)]=best[1]
  bases={}; tails={}
  for row in sorted(set(map(int,queries))|set(within.values())|set(cross.values())):
   r=int(y[row]); _,p=search[r].search(np.ascontiguousarray(x[row:row+1]),31)
   local=np.asarray(x[rr[r][p[0,1:]]]-x[row],np.float64); _,sv,vt=np.linalg.svd(local,full_matrices=False)
   bases[row]=vt.T[:,:8]; e=sv**2; tails[row]=float(e[8:].sum()/max(e.sum(),1e-12))
  td=np.mean([proj(bases[int(q)],bases[cross[int(q)]])-proj(bases[int(q)],bases[within[int(q)]]) for q in queries])
  out[seed]=(float(np.mean(y[nn]==y[:,None])),float(np.mean(other/np.maximum(same,1e-12))),float(td),float(np.mean([tails[int(q)] for q in queries])))
 return out

def residual_eta(x,left,right,actions,labels,coef,mean,scale,k,chunk):
 d=x.shape[1]; sums={s:np.zeros((k,d)) for s in labels}; counts={s:np.zeros(k,int) for s in labels}; total=np.zeros(d); sq=0.; n=0
 for b in range(0,len(left),chunk):
  e=min(b+chunk,len(left)); a=(actions[b:e]-mean)/scale; design=np.c_[np.ones(len(a)),a]
  dz=x[right[b:e]].astype(np.float64)-x[left[b:e]].astype(np.float64); z=dz-design@coef
  total+=z.sum(0); sq+=float(np.square(z).sum()); n+=len(z)
  for s,y in labels.items():
   yy=y[left[b:e]]
   for r in range(k):
    m=yy==r; counts[s][r]+=m.sum(); sums[s][r]+=z[m].sum(0)
 gm=total/n; denom=sq-n*float(gm@gm); out={}
 for s in labels:
  means=sums[s]/counts[s][:,None]; out[s]=float(np.sum(counts[s]*np.sum((means-gm)**2,axis=1))/max(denom,1e-12))
 return out

def velocity_eta(x,left,right,labels,k,chunk):
 d=x.shape[1]; sums={s:np.zeros((k,d)) for s in labels}; counts={s:np.zeros(k,int) for s in labels}; total=np.zeros(d); sq=0.; n=0
 for b in range(0,len(left),chunk):
  e=min(b+chunk,len(left)); dz=x[right[b:e]].astype(np.float64)-x[left[b:e]].astype(np.float64)
  total+=dz.sum(0); sq+=float(np.square(dz).sum()); n+=len(dz)
  for s,y in labels.items():
   yy=y[left[b:e]]
   for r in range(k):
    m=yy==r; counts[s][r]+=m.sum(); sums[s][r]+=dz[m].sum(0)
 gm=total/n; denom=sq-n*float(gm@gm); out={}
 for s in labels:
  means=sums[s]/counts[s][:,None]; out[s]=float(np.sum(counts[s]*np.sum((means-gm)**2,axis=1))/max(denom,1e-12))
 return out

def task_rows(repo,model,t,ks,seeds,a):
 m4=json.loads(gate(repo,model,t,4).read_text()); x,ids=rg.load_unique(Path(m4["cache_stats"]["cache"]),a.frameskip); data=rg.resolve_data_file(Path(m4["data_file"]))
 with h5py.File(data,"r",swmr=True) as h:
  ek="episode_idx" if "episode_idx" in h else "ep_idx"; groups=np.asarray(h[ek][ids],np.int64)
 audit_path=a.audit_dir/f"{t}.npz"
 if not audit_path.exists():
  a.audit_dir.mkdir(parents=True,exist_ok=True); k4labels={s:rg.load_labels(root(repo,model,t,4,s)/"cluster_labels.npz",ids) for s in seeds}; used=np.zeros(len(x),bool)
  for s in seeds: used[_sample_landmarks(len(x),min(20000,len(x)),s,groups)]=True
  eligible=np.flatnonzero(~used); pick=eligible[_sample_landmarks(len(eligible),min(20000,len(eligible)),20260824,groups[eligible])]
  zp=root(repo,model,t,4,0)/"partition/zscore_params.npz"; z=np.load(zp); ax0=np.asarray((x[pick]-z["mean"])/z["scale"],np.float32); ax0/=np.maximum(np.linalg.norm(ax0,axis=1,keepdims=True),1e-12)
  np.savez(audit_path,x=ax0,sample_ids=ids[pick],groups=groups[pick])
 with np.load(audit_path,allow_pickle=False) as z: ax=np.asarray(z["x"],np.float32); aids=np.asarray(z["sample_ids"],np.int64)
 apos=np.searchsorted(ids,aids); assert np.array_equal(ids[apos],aids); rows=[]
 left,right,actions=rg.transition_rows(ids,data,1); action_by=np.zeros((len(ids),actions.shape[1])); action_by[left]=actions
 for k in ks:
  labels={s:rg.load_labels(root(repo,model,t,k,s)/"cluster_labels.npz",ids) for s in seeds}; ays={s:labels[s][apos] for s in seeds}
  am=audit_metrics(ax,ays,k); fitted,residual,sqrt_cov,amean,ascale=rg.sufficient_stats(x,left,right,actions,labels,k,a.ridge,a.chunk)
  penalty=np.eye(actions.shape[1]+1)*a.ridge; penalty[0,0]=0; aa=(actions-amean)/ascale
  design_all=np.c_[np.ones(len(aa)),aa]; xtx=design_all.T@design_all; xty=design_all.T@(x[right].astype(np.float64)-x[left].astype(np.float64)); gcoef=np.linalg.solve(xtx+penalty,xty); reference_moment=xtx/len(left)
  reta=residual_eta(x,left,right,actions,labels,gcoef,amean,ascale,k,a.chunk); veta=velocity_eta(x,left,right,labels,k,a.chunk)
  fractions,eig=meta(root(repo,model,t,k,0)); fractions=(np.bincount(labels[0],minlength=k)/len(labels[0]) if fractions is None else fractions); pp,op=protos(root(repo,model,t,k,0)); p=np.load(pp); own=np.load(op); p/=np.maximum(np.linalg.norm(p,axis=1,keepdims=True),1e-12)
  dist=1-p@p.T; np.fill_diagonal(dist,np.inf); same=own[:,None]==own[None,:]; pr=float(np.where(~same,dist,np.inf).min(1).mean()/np.where(same,dist,np.inf).min(1).mean())
  for s in seeds:
   y=labels[s]; valid=(ids[10:]-ids[:-10]==10)&(groups[10:]==groups[:-10]); flow=float(np.mean(y[:-10][valid]==y[10:][valid]))
   pairs=[]; jacs=[]; cents=[]; bounds=[]; ay=ays[s]; rr={r:np.flatnonzero(ay==r) for r in range(k)}; inds={}
   for r in range(k): q=faiss.IndexFlatIP(ax.shape[1]); q.add(np.ascontiguousarray(ax[rr[r]])); inds[r]=q
   for i,j in combinations(range(k),2):
    ci,cj=fitted[s][i][0],fitted[s][j][0]; di=ci[0]-cj[0]; ds=sqrt_cov@(ci[1:]-cj[1:]); pairs.append(float((np.sum(di*di)+np.sum(ds*ds))/residual)); jacs.append(rg.jacobian_metrics(ci,cj,sqrt_cov))
    cents.append(float(np.linalg.norm(ax[rr[i]].mean(0)-ax[rr[j]].mean(0)))); sij,_=inds[j].search(np.ascontiguousarray(ax[rr[i]]),1); sji,_=inds[i].search(np.ascontiguousarray(ax[rr[j]]),1); bounds.append(.5*(np.mean(1-sij[:,0])+np.mean(1-sji[:,0])))
   pairs=np.asarray(pairs); jacs=np.asarray(jacs)
   affine=sum((fitted[s][r][1]/len(left))*float(np.trace((fitted[s][r][0]-gcoef).T@reference_moment@(fitted[s][r][0]-gcoef))) for r in range(k))/residual
   z=(ax-ax.mean(0))/(ax.std(0)+1e-8); z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12); nn=faiss.IndexFlatIP(z.shape[1]); nn.add(np.ascontiguousarray(z)); _,nb=nn.search(np.ascontiguousarray(z),31); mixed=np.any(ay[nb[:,1:]]!=ay[:,None],axis=1)
   br=apos[mixed]; nxt=np.searchsorted(ids,ids[br]+1); ok=(nxt<len(ids))&(ids[nxt]==ids[br]+1); br=br[ok]
   bcoef,bcov=rg.fit_subset(x,ids,action_by,labels[s],br,k,a.ridge,1); bb=min(rg.jacobian_metrics(bcoef[i],bcoef[j],bcov)[3] for i,j in combinations(range(k),2))
   if model=="subjepa":
    kb=pd.read_csv(repo/"experiments/control_matrix/assets/gate_sensitivity_subjepa/k_behavior.csv"); gr=kb[(kb.task==t)&(kb.K==k)].iloc[0]; check1=float(gr.safety_margin+.5); check2=float(gr.prominence_margin)
   else:
    ga=json.loads(gate(repo,model,t,k).read_text())["method_metadata"]["automatic_gate"]; check1=float(ga["retained_safety_fraction"]); check2=float(ga["robust_residual_gap"]/ga["background_threshold"]-1.)
   rows.append(dict(task=t,num_clusters=k,partition_seed=s,normalized_cluster_entropy=float(-(fractions*np.log(fractions)).sum()/np.log(k)),eigengap_after_k=eig,prototype_distance_ratio=pr,
    knn_purity=am[s][0],margin_radius_ratio_mean=am[s][1],tangent_contrast_mean_d8=am[s][2],curvature_tail_mean_d8=am[s][3],flow_persistence_h10=flow,latent_velocity_eta2=veta[s],action_residual_velocity_eta2=reta[s],
    affine_response_contrast_ratio=float(affine),minimum_pairwise_response=float(pairs.min()),pairwise_uniformity_min_over_mean=float(pairs.min()/pairs.mean()),response_centroid_spearman=float(spearmanr(pairs,cents).statistic) if len(pairs)>1 else np.nan,response_boundary_spearman=float(spearmanr(pairs,bounds).statistic) if len(pairs)>1 else np.nan,boundary_min_jacobian_bures_distance=bb,jacobian_bures_distance=float(jacs[:,3].min()),jacobian_cosine_distance=float(jacs[:,0].min()),jacobian_log_scale_distance=float(jacs[:,1].min()),jacobian_subspace_chordal_distance=float(jacs[:,2].min()),check1_retained_safety_fraction=check1,check2_prominence_ratio=check2))
  print(f"completed {t} K={k}",flush=True)
 return rows

def fit(v,y):
 v=np.asarray(v,float); y=np.asarray(y)=="regional"; u=np.unique(v); span=max(float(np.ptp(u)),1.); cand=np.r_[u[0]-span,(u[:-1]+u[1:])/2,u[-1]+span]; best=None
 for d in ("higher","lower"):
  for th in cand:
   p=v>th if d=="higher" else v<th; bal=.5*(np.mean(p[y])+np.mean(~p[~y])); key=(bal,np.mean(p==y),np.min(np.abs(v-th))/span,d=="higher",-th)
   if best is None or key>best[0]: best=(key,d,float(th))
 return best[1],best[2]

def benchmark(repo,raw,out,model):
 screen=pd.read_csv(repo/"experiments/control_matrix/assets/lewm_k4_geometry_screen/metric_screen_summary.csv").set_index("metric"); lewm=pd.read_csv(repo/"experiments/control_matrix/assets/lewm_k4_geometry_screen/frozen_bures_gate_validation.csv"); l1=lewm[lewm.num_clusters.eq(4)].set_index("task")
 targets=(pd.read_csv(repo/"experiments/control_matrix/assets/subjepa_k_geometry_screen/jacobian_fixed_k_validation.csv") if model=="subjepa" else lewm); l2=targets[targets.num_clusters.isin(sorted(raw.num_clusters.unique()))][["task","num_clusters","global_mean_percent","regional_mean_percent","delta_regional_minus_global_pp","point_estimate_winner"]]
 scores=raw.groupby(["task","num_clusters"],as_index=False)[list(METRICS)].mean(numeric_only=True); details=l2.merge(scores,on=["task","num_clusters"],validate="one_to_one"); policies=[]; bth=float(lewm.frozen_bures_threshold.dropna().unique()[0])
 for metric in METRICS:
  if metric.startswith("check"):
   vals=[]
   for t in TASKS:
    g=json.loads(gate(repo,"lewm",t,4).read_text())["method_metadata"]["automatic_gate"]; vals.append(float(g["retained_safety_fraction"] if metric.startswith("check1") else g["robust_residual_gap"]/g["background_threshold"]-1.))
  else: vals=[float(screen.loc[metric,t]) for t in TASKS]
  labels=[l1.loc[t,"point_estimate_winner"] for t in TASKS]
  if metric=="jacobian_bures_distance": d,th,src="higher",bth,"predeclared_existing"
  elif metric=="check1_retained_safety_fraction": d,th,src="higher",.5,"predeclared_existing"
  elif metric=="check2_prominence_ratio": d,th,src="higher",0.,"predeclared_R_minus_Tbg"
  else: d,th=fit(vals,labels); src="K4_only_balanced_accuracy"
  pred=np.asarray(vals)>th if d=="higher" else np.asarray(vals)<th; policies.append(dict(metric=metric,direction=d,threshold=th,threshold_source=src,layer1_correct=int(np.sum(pred==(np.asarray(labels)=="regional"))),layer1_total=4,layer1_accuracy=float(np.mean(pred==(np.asarray(labels)=="regional")))))
 policies=pd.DataFrame(policies); preds=[]; summary=[]
 for p in policies.itertuples(index=False):
  av=details[p.metric].notna(); reg=details[p.metric]>p.threshold if p.direction=="higher" else details[p.metric]<p.threshold; hit=reg.map({True:"regional",False:"global"}).eq(details.point_estimate_winner)&av
  for i,r in details.iterrows(): preds.append(dict(metric=p.metric,task=r.task,num_clusters=r.num_clusters,score=r[p.metric],direction=p.direction,threshold=p.threshold,predicted_branch=("regional" if reg[i] else "global") if av[i] else "abstain",point_estimate_winner=r.point_estimate_winner,correct=bool(hit[i]) if av[i] else False,delta_regional_minus_global_pp=r.delta_regional_minus_global_pp))
  cov=int(av.sum()); h=int(hit.sum()); total=len(details); summary.append(dict(metric=p.metric,layer2_correct=h,layer2_covered=cov,layer2_total=total,layer2_accuracy=h/cov if cov else np.nan,full_grid_accuracy=h/total,direction=p.direction,threshold=p.threshold,layer1_accuracy=p.layer1_accuracy,threshold_source=p.threshold_source))
 summary=pd.DataFrame(summary).sort_values(["full_grid_accuracy","layer2_accuracy","layer2_covered"],ascending=False); preds=pd.DataFrame(preds); policies.to_csv(out/"layer1_frozen_policies.csv",index=False); details.to_csv(out/"layer2_metric_scores.csv",index=False); preds.to_csv(out/"layer2_predictions.csv",index=False); summary.to_csv(out/"layer2_selection_accuracy.csv",index=False); return summary

def main():
 a=args(); repo=a.repo.resolve(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True); faiss.omp_set_num_threads(a.cpu_threads); rows=[]
 for t in filter(None,a.tasks.split(",")): rows+=task_rows(repo,a.model,t,ints(a.clusters),ints(a.partition_seeds),a)
 raw=pd.DataFrame(rows); raw.to_csv(out/"layer2_metric_scores_by_seed.csv",index=False); summary=benchmark(repo,raw,out,a.model); b=summary[summary.metric.eq("jacobian_bures_distance")].iloc[0]
 manifest=dict(schema_version=1,analysis_name=f"{a.model} frozen 22-criterion benchmark",repository_commit=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),model=a.model,criterion_count=len(METRICS),development_model="lewm",development_num_clusters=4,validation_num_clusters=list(ints(a.clusters)),partition_seeds=list(ints(a.partition_seeds)),ridge=a.ridge,target="point-estimate winner of partition-seed-averaged Regional versus Global",threshold_leakage_check="Sub-JEPA outcomes are never passed to fit",bures_layer2_correct=int(b.layer2_correct),bures_layer2_total=int(b.layer2_total),bures_layer2_accuracy=float(b.full_grid_accuracy),files={})
 for p in sorted(out.glob("*.csv")): manifest["files"][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 (out/"benchmark_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n"); assert len(METRICS)==len(summary)==22; print(summary.to_string(index=False)); print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
