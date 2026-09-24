import numpy as np,glob,sys,json
from scipy.io import loadmat
from scipy.signal import butter,sosfilt
names=["c4","rpa","f8","p8","f4","p4","fp2","o2","cz","pz","fz","o1","fp1","p3","f3","p7","f7","lpa","c3","c7","c8"]
ic3,ic4=names.index('c3'),names.index('c4')
out=open(sys.argv[2],'a')
for p in sorted(glob.glob(sys.argv[1]+'/*.mat'))[int(sys.argv[3]):int(sys.argv[4])]:
  m=loadmat(p,simplify_cells=True);E=m['EEG']
  X=np.asarray(E['Raw'],float);S=np.asarray(E['States']).astype(int).ravel()
  X=X-np.median(X,0)
  bp=lambda lo,hi:sosfilt(butter(4,[lo,hi],'bandpass',fs=250,output='sos'),X,axis=0)**2
  A=bp(81.5,85.5).mean(1); M=bp(8,13); Bt=bp(15,25)
  d=np.flatnonzero(np.diff(np.r_[-1,S])!=0); ends=np.r_[d[1:],len(S)]
  R=[];T=[]
  for a,b in zip(d,ends):
    c=S[a]
    if c==0: continue
    early=slice(a+125,a+625); late=slice(a+1250,b)
    f=lambda Z,s:np.log(Z[s].mean(0)+1e-9)
    T.append(dict(c=c,art_late=np.log(A[late].mean()+1e-9),art_early=np.log(A[early].mean()+1e-9),
      mu_e=f(M,early)[[ic3,ic4]],mu_l=f(M,late)[[ic3,ic4]]))
  rest=[t for t in T if t['c']==1]; mi=[t for t in T if t['c']>1]
  r_mu_e=np.mean([t['mu_e'] for t in rest],0); r_mu_l=np.mean([t['mu_l'] for t in rest],0); r_art=np.mean([t['art_late'] for t in rest])
  art=np.array([t['art_late']-r_art for t in mi])*10/np.log(10)
  mul=np.array([ (t['mu_l']-r_mu_l).mean() for t in mi])*10/np.log(10)
  mue=np.array([ (t['mu_e']-r_mu_e).mean() for t in mi])*10/np.log(10)
  hi=art>6; lo=art<2
  res=dict(s=p.split('/')[-1][:-4],n_hi=int(hi.sum()),n_lo=int(lo.sum()),
    ERDmu_late_hiArt=round(float(mul[hi].mean()),2) if hi.any() else None,
    ERDmu_late_loArt=round(float(mul[lo].mean()),2) if lo.any() else None,
    ERDmu_early=round(float(mue.mean()),2), ERDmu_late=round(float(mul.mean()),2),
    corr_art_mu=round(float(np.corrcoef(art,mul)[0,1]),2))
  out.write(json.dumps(res)+'\n');out.flush()
