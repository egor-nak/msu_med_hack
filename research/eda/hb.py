import numpy as np,glob,sys
from scipy.io import loadmat
from scipy.signal import welch
D=sys.argv[1];r1=[];r2=[];r3=[]
for p in sorted(glob.glob(D+'/*.mat')):
  m=loadmat(p,simplify_cells=True)
  if 'NIRS' not in m: continue
  N=m['NIRS'];fs=float(N['Frq'])
  if fs<10: continue
  O=np.asarray(N['HbO'],float);R=np.asarray(N['HbR'],float)
  f,po=welch(O-O.mean(0),fs=fs,nperseg=512,axis=0);_,pr=welch(R-R.mean(0),fs=fs,nperseg=512,axis=0)
  card=(f>0.8)&(f<1.8); slow=(f>0.01)&(f<0.2); mayer=(f>0.07)&(f<0.13)
  r1.append(np.median(po[card].sum(0)/pr[card].sum(0)))
  r2.append(np.median(po[slow].sum(0)/pr[slow].sum(0)))
  # cardiac peak prominence in each: peak / neighbours
  prom=lambda P:np.median(P[card].max(0)/np.median(P[(f>2)&(f<4)],0))
  r3.append((prom(po),prom(pr)))
r3=np.array(r3)
print('n',len(r1),'cardiac power HbO/HbR median',np.round(np.median(r1),2),'IQR',np.round(np.percentile(r1,[25,75]),2))
print('slow power HbO/HbR median',np.round(np.median(r2),2),np.round(np.percentile(r2,[25,75]),2))
print('cardiac peak prominence HbO',np.round(np.median(r3[:,0]),1),'HbR',np.round(np.median(r3[:,1]),1))
