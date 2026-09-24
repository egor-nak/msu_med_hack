import numpy as np,glob,sys
from scipy.io import loadmat
from scipy.signal import butter,sosfilt
names=["c4","rpa","f8","p8","f4","p4","fp2","o2","cz","pz","fz","o1","fp1","p3","f3","p7","f7","lpa","c3","c7","c8"]
I=lambda n:names.index(n)
def auc(a,b):
  r=np.argsort(np.argsort(np.r_[a,b]))+1
  return (r[:len(a)].sum()-len(a)*(len(a)+1)/2)/(len(a)*len(b))
R={k:[] for k in ['heog_f7f8','heog_f7f8_early','alpha_o1o2','mu_c3c4','beta_c3c4','mu_c3c4_early','hf_c7c8']}
for p in sorted(glob.glob(sys.argv[1]+'/*.mat'))[int(sys.argv[2]):int(sys.argv[3])]:
  m=loadmat(p,simplify_cells=True);E=m['EEG']
  X=np.asarray(E['Raw'],float);S=np.asarray(E['States']).astype(int).ravel()
  X=X-np.median(X,0)
  lp=sosfilt(butter(2,[0.1,3],'bandpass',fs=250,output='sos'),X,axis=0)
  bp=lambda lo,hi:sosfilt(butter(4,[lo,hi],'bandpass',fs=250,output='sos'),X,axis=0)**2
  A=bp(8,13);Bt=bp(15,28);H=bp(30,45)
  d=np.flatnonzero(np.diff(np.r_[-1,S])!=0);ends=np.r_[d[1:],len(S)]
  F={k:{2:[],3:[]} for k in R}
  for a,b in zip(d,ends):
    c=S[a]
    if c not in (2,3): continue
    w=slice(a+250,b);e=slice(a,a+500)
    heog=lambda s:(lp[s,I('f7')]-lp[s,I('f8')]).mean()-(lp[a-300:a,I('f7')]-lp[a-300:a,I('f8')]).mean()
    F['heog_f7f8'][c].append(heog(w)); F['heog_f7f8_early'][c].append(heog(e))
    lr=lambda Z,s,l,r:np.log(Z[s,I(l)].mean())-np.log(Z[s,I(r)].mean())
    F['alpha_o1o2'][c].append(lr(A,w,'o1','o2'))
    F['mu_c3c4'][c].append(lr(A,w,'c3','c4'));F['beta_c3c4'][c].append(lr(Bt,w,'c3','c4'))
    F['mu_c3c4_early'][c].append(lr(A,slice(a+125,a+625),'c3','c4'))
    F['hf_c7c8'][c].append(lr(H,w,'c7','c8'))
  for k in R: R[k].append(auc(np.array(F[k][2]),np.array(F[k][3])))
for k,v in R.items():
  v=np.array(v);dev=np.abs(v-.5)
  print(f'{k:18s} n={len(v)} median AUC={np.median(v):.3f}  median|AUC-0.5|={np.median(dev):.3f}  frac |AUC-.5|>0.25: {np.mean(dev>0.25):.2f}')
np.save(sys.argv[4],R,allow_pickle=True)
