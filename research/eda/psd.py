import sys,numpy as np,glob
from scipy.io import loadmat
from scipy.signal import welch,butter,sosfilt
names=["c4","rpa","f8","p8","f4","p4","fp2","o2","cz","pz","fz","o1","fp1","p3","f3","p7","f7","lpa","c3","c7","c8"]
res=[]
files=sorted(glob.glob('*.mat'))
out={}
for p in files:
  m=loadmat(p,simplify_cells=True);E=m['EEG']
  X=np.asarray(E['Raw'],float);S=np.asarray(E['States']).astype(int).ravel()
  X=X-np.median(X,0)
  sos=butter(2,1,'hp',fs=250,output='sos');X=sosfilt(sos,X,axis=0)
  P={}
  for c in (1,2,3):
    idx=np.flatnonzero(S==c)
    # split into 1s epochs fully inside class
    d=np.flatnonzero(np.diff(np.r_[0,(S==c).astype(int),0])!=0).reshape(-1,2)
    ps=[]
    for a,b in d:
      a+=250 # skip first second
      for s in range(a,b-500,250):
        f,pp=welch(X[s:s+500],fs=250,nperseg=250,axis=0);ps.append(pp)
    P[c]=np.median(np.array(ps),0)
  out[p]=(f,P)
  fi=lambda hz:int(round(hz))
  r=lambda c,ch,hz:10*np.log10(P[c][fi(hz),names.index(ch)]/P[1][fi(hz),names.index(ch)])
  line=p[:-4][-3:]+' '+p[:10]
  for hz in (10,20,30,40,50,70,80,100,110):
    line+=f' | {hz}Hz L/R-vs-rest C3:{r(2,"c3",hz):+.1f}/{r(3,"c3",hz):+.1f} C4:{r(2,"c4",hz):+.1f}/{r(3,"c4",hz):+.1f}'
  print(line,flush=True)
np.save('/tmp/x.npy',0)
