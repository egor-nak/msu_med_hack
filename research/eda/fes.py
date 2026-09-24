import numpy as np,glob
from scipy.io import loadmat
from scipy.signal import welch,butter,sosfilt
names=["c4","rpa","f8","p8","f4","p4","fp2","o2","cz","pz","fz","o1","fp1","p3","f3","p7","f7","lpa","c3","c7","c8"]
for p in sorted(glob.glob('*.mat')):
  m=loadmat(p,simplify_cells=True);E=m['EEG']
  X=np.asarray(E['Raw'],float);S=np.asarray(E['States']).astype(int).ravel()
  X=X-np.median(X,0);X=sosfilt(butter(2,1,'hp',fs=250,output='sos'),X,axis=0)
  P={}
  for c in (1,2,3):
    d=np.flatnonzero(np.diff(np.r_[0,(S==c).astype(int),0])!=0).reshape(-1,2)
    ps=[welch(X[a+500:b],fs=250,nperseg=1000,axis=0)[1] for a,b in d]
    f=welch(X[:2000],fs=250,nperseg=1000,axis=0)[0]
    P[c]=np.mean(ps,0)
  for c in (2,3):
    R=10*np.log10(P[c]/P[1])
    R[f<5]=0
    k=np.argsort(R.max(1))[::-1][:6]
    print(p[-7:-4],'class',c,'top ratio freqs:',[(round(f[i],2),round(R[i].max(),1),names[R[i].argmax()]) for i in k])
  # absolute spectral peaks rest (line / FES)
  lp=10*np.log10(P[1].mean(1)); base=np.convolve(lp,np.ones(41)/41,'same')
  pk=np.argsort(lp-base)[::-1][:8]
  print('   rest spectral peaks:',[(round(f[i],2),round((lp-base)[i],1)) for i in pk])
