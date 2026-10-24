import sys, collections, pandas as pd, numpy as np
R=sys.argv[1]
ex=collections.defaultdict(list); strand={}
for l in open(f'{R}/annotation.chr7.gtf'):
    if l.startswith('#'): continue
    f=l.split('\t')
    if f[2]!='exon': continue
    tid=f[8].split('transcript_id "')[1].split('"')[0]; ex[tid].append((int(f[3])-1,int(f[4]))); strand[tid]=f[6]
def gpos(tid,i):
    e=sorted(ex[tid],reverse=strand[tid]=='-')
    for s,t in e:
        L=t-s
        if i<L: return s+i if strand[tid]=='+' else t-1-i
        i-=L
idx=pd.read_parquet('data/p6/txcache/fc2d_k562/index.parquet'); idx=idx[idx.chrom=='chr7']
mane=set(); 
for t in idx.transcript_id:
    for s,e in ex.get(t,[]): mane.update(range(s,e))
def load(p):
    d=pd.read_csv(p,sep='\t'); d['g']=[gpos(t,s) for t,s in zip(d.Tid,d.Start)]; return d
real=load(f'{R}/real/K562_TIS_predict.txt'); pred=load(f'{R}/predicted/K562_TIS_predict.txt')
for name,d in [('real',real),('pred',pred)]:
    d['inMANE']=d.g.isin(mane); d['maneTid']=d.Tid.isin(set(idx.transcript_id))
    t=d.groupby(d.TisType.str.split(':').str[0]).agg(n=('g','size'),inMANEexon=('inMANE','mean'),onMANEtx=('maneTid','mean'))
    print(name); print(t.round(2).to_string())
r5=real[real.TisType.str.startswith("5'UTR")]; ps=set(zip(pred.g,pred.StartCodon))
r5=r5.assign(hit=[g in set(pred.g) for g in r5.g])
print("real 5'UTR recall by inMANE exon:", r5.groupby('inMANE').hit.agg(['size','mean']).round(2).to_dict())
print('--- unique sites')
for lab,key in [('pos',lambda d:set(d.g)),('pos+codon',lambda d:set(zip(d.g,d.StartCodon))),('pos+type',lambda d:set(zip(d.g,d.TisType.str.split(':').str[0])))]:
    r5=real[real.TisType.str.startswith("5'UTR")]; P=key(pred); R5=key(r5)
    print(lab,'real 5UTR unique',len(R5),'recall',round(len(R5&P)/len(R5),3))
r5=real[real.TisType.str.startswith("5'UTR")].drop_duplicates('g')
pt=pred.drop_duplicates('g').set_index('g').TisType.str.split(':').str[0]
print('pred type at real 5UTR sites:', pt.reindex(r5.g).fillna('MISSED').value_counts().to_dict())
