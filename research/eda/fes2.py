import numpy as np,glob,sys,json
from scipy.io import loadmat
from scipy.signal import butter,sosfilt
names=["c4","rpa","f8","p8","f4","p4","fp2","o2","cz","pz","fz","o1","fp1","p3","f3","p7","f7","lpa","c3","c7","c8"]
D='/'+sys.argv[1].strip('/')
out=open(sys.argv[2],'w')
def auc(a,b):
  a=np.asarray(a);b=np.asarray(b)
  r=np.argsort(np.argsort(np.r_[a,b]))+1
  return (r[:len(a)].sum()-len(a)*(len(a)+1)/2)/(len(a)*len(b))
for p in sorted(glob.glob(D+'/*.mat')):
  m=loadmat(p,simplify_cells=True);E=m['EEG']
  X=np.asarray(E['Raw'],float);S=np.asarray(E['States']).astype(int).ravel();Bk=np.asarray(E['Blocks']).astype(int).ravel()
  X=X-np.median(X,0)
  Y=sosfilt(butter(4,[81.5,85.5],'bandpass',fs=250,output='sos'),X,axis=0)
  mu=sosfilt(butter(4,[8,30],'bandpass',fs=250,output='sos'),X,axis=0)
  pw=np.log(Y**2+1e-9); 
  # 1s window log power
  cs=np.cumsum(np.r_[np.zeros((1,21)),Y**2],0); csm=np.cumsum(np.r_[np.zeros((1,21)),mu**2],0)
  d=np.flatnonzero(np.diff(np.r_[-1,S])!=0); ends=np.r_[d[1:],len(S)]
  rows=[]
  tc={1:[],2:[],3:[]}
  for a,b in zip(d,ends):
    c=S[a]
    if c==0: continue
    seg=[np.log((cs[t+250]-cs[t])/250+1e-9) for t in range(a,b-250+1,62)]
    tc[c].append(np.array([np.log((cs[a+k*125+250]-cs[a+k*125])/250+1e-9) for k in range(int((b-a-250)/125))]))
    rows.append((c,Bk[a],np.mean(seg,0)))
  C=np.array([r[0] for r in rows]);F=np.array([r[2] for r in rows])
  # AUC per channel: MI vs rest, L vs R
  a_mi=[auc(F[C>1,j],F[C==1,j]) for j in range(21)]
  a_lr=[auc(F[C==2,j],F[C==3,j]) for j in range(21)]
  # lateralization index c7-c8, c3-c4
  li=F[:,names.index('c7')]-F[:,names.index('c8')]
  res=dict(s=p.split('/')[-1][:-4],auc_mi_max=round(max(a_mi),3),ch_mi=names[int(np.argmax(a_mi))],
    auc_lr_extreme=round(max(a_lr,key=lambda v:abs(v-.5)),3),ch_lr=names[int(np.argmax(np.abs(np.array(a_lr)-.5)))],
    auc_lr_c7c8=round(auc(li[C==2],li[C==3]),3),
    db_mi_vs_rest_max=round(float(10/np.log(10)*(np.median(F[C>1],0)-np.median(F[C==1],0)).max()),1),
    frac_MI_trials_gt6dB=round(float(np.mean((F[C>1].max(1)-np.median(F[C==1],0).max())*10/np.log(10)>6)),2),
    tc_MI=[round(float(v),1) for v in (10/np.log(10)*(np.mean([t[:14].max(1) if False else t[:14].mean(1) for t in tc[2]+tc[3] if len(t)>=14],0)-np.mean([t[:14].mean(1) for t in tc[1] if len(t)>=14],0)))])
  out.write(json.dumps(res)+'\n');out.flush()
