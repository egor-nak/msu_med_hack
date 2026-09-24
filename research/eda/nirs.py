import numpy as np,glob,sys
from scipy.io import loadmat
D=sys.argv[1]
acc={c:{'o':[],'r':[]} for c in (1,2,3)}; corr=[]; auc_lr=[]
def auc(a,b):
  r=np.argsort(np.argsort(np.r_[a,b]))+1
  return (r[:len(a)].sum()-len(a)*(len(a)+1)/2)/(len(a)*len(b))
for p in sorted(glob.glob(D+'/*.mat')):
  m=loadmat(p,simplify_cells=True)
  if 'NIRS' not in m: continue
  N=m['NIRS'];fs=float(N['Frq'])
  if fs<10: continue
  O=np.asarray(N['HbO'],float);R=np.asarray(N['HbR'],float);S=np.asarray(N['States']).astype(int).ravel()
  sd=np.median(np.abs(np.diff(O,axis=0)),0)+1e-12
  # global HbO-HbR corr of detrended signals (per channel, 0.01-0.2 Hz approx by differencing)
  corr.append(np.median([np.corrcoef(np.diff(O[:,j]),np.diff(R[:,j]))[0,1] for j in range(18)]))
  d=np.flatnonzero(np.diff(np.r_[-1,S])!=0)
  L=[];Rr=[]
  for a in d:
    c=S[a]
    if c==0 or a<int(2*fs) or a+int(18*fs)>len(S): continue
    base=O[a-int(2*fs):a].mean(0); baseR=R[a-int(2*fs):a].mean(0)
    eo=(O[a-int(2*fs):a+int(18*fs)]-base); er=(R[a-int(2*fs):a+int(18*fs)]-baseR)
    # normalize per channel scale
    s=np.std(O,0)+1e-12; sr=np.std(R,0)+1e-12
    acc[c]['o'].append(eo/s); acc[c]['r'].append(er/sr)
    if c>1:
      late=slice(int(2*fs)+int(5*fs),int(2*fs)+int(10*fs))
      li=(eo[late,8:16].mean()-eo[late,0:8].mean())/np.mean(s)  # left-hemi minus right-hemi
      (L if c==2 else Rr).append(li)
  if len(L)>3 and len(Rr)>3: auc_lr.append(auc(np.array(Rr),np.array(L)))
t=np.arange(-2,18,1/12.5)
for c in (1,2,3):
  o=np.mean(acc[c]['o'],0); r=np.mean(acc[c]['r'],0)
  print('class',c,'n',len(acc[c]['o']))
  for tt in (0,2,4,6,8,10,12,14,16):
    k=int((tt+2)*12.5)
    print(f'  t={tt:2d}s HbO L-hemi {o[k,8:16].mean():+.3f} R-hemi {o[k,0:8].mean():+.3f} | HbR L {r[k,8:16].mean():+.3f} R {r[k,0:8].mean():+.3f}')
print('median diff-corr HbO/HbR per session', np.round(np.percentile(corr,[10,50,90]),2))
print('LR AUC (right-hand MI should give higher left-hemi HbO): median',np.median(auc_lr),'IQR',np.percentile(auc_lr,[25,75]),'n',len(auc_lr))
