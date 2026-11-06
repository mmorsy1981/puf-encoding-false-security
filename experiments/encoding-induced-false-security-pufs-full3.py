"""Noisy full: OB-Fig3 + LFSR-Eq7 10x3 at nz=0.1; RSO m=32/N=200k noisy 5x2."""
import time, csv, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.linear_model import LogisticRegression
t0 = time.time()
def to_pm(C01): return (2*C01.astype(np.int8) - 1)
def labels(puf, C01): return ((puf.eval(to_pm(C01)).reshape(-1) + 1)//2).astype(int)
def pfeat(C):
    S = 1-2*C.astype(float); X = np.ones((len(C),65))
    for i in range(64): X[:,i]=np.prod(S[:,i:],axis=1)
    return X
def lr_acc(Xtr,ytr,Xte,yte,seed=0):
    if len(np.unique(ytr))<2: return float((yte==np.bincount(yte).argmax()).mean())
    c=LogisticRegression(max_iter=1000,solver='liblinear',random_state=seed); c.fit(Xtr.astype(float),ytr); return float(c.score(Xte.astype(float),yte))
OB_POS=[[0,1,2],[0,1,63],[0,62,63],[61,62,63]]
OB_VAL=np.array([[0,1,0],[1,0,1],[0,1,0],[1,0,1]],dtype=np.uint8) if False else np.array([[0,1,0],[1,0,1],[0,1,0],[1,0,1]],dtype=np.uint8)
OB_SR=np.array([[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,1,0]],dtype=np.uint8)
def ob_full(Cob,i):
    N=len(Cob);C=np.zeros((N,64),dtype=np.uint8)
    for t in range(N):
        pos=sorted(OB_POS[i]);vals=OB_VAL[i];src=0;p=0
        for dst in range(64):
            if p<3 and dst==pos[p]:C[t,dst]=vals[p];p+=1
            else:C[t,dst]=Cob[t,src];src+=1
    return C
def galois(s,cl,taps=(0,1,3,4)):
    s=s.copy()
    for _ in range(cl):
        fb=s[-1]
        for t in taps:s[t]^=fb
        s=np.roll(s,1);s[0]=fb
    return s
def lfsr_int(C68,base=10):
    N=len(C68);o=np.zeros((N,64),dtype=np.uint8)
    for t in range(N):
        k=int(C68[t,64:].dot([8,4,2,1]))*base
        s=C68[t,:64].copy()
        for _ in range(k):
            fb=s[-1]
            for tt in (0,1,3,4):s[tt]^=fb
            s=np.roll(s,1);s[0]=fb
        o[t]=s
    return o
rows=[]
for inst in range(10):
    for asd in [0,1,2]:
        pufs=[ArbiterPUF(n=64,seed=300+inst*10+j,noisiness=0.1) for j in range(4)]
        rng=np.random.default_rng(8000+inst*10+asd)
        Cob=rng.integers(0,2,size=(20000,61)).astype(np.uint8);ii=rng.integers(0,4,size=20000)
        Cte=rng.integers(0,2,size=(5000,61)).astype(np.uint8);iii=rng.integers(0,4,size=5000)
        Rob=np.zeros((20000,4),dtype=int);Rte=np.zeros((5000,4),dtype=int)
        for i in range(4):
            m=ii==i
            if m.sum():Rob[m]=np.column_stack([labels(p,ob_full(Cob[m],i)) for p in pufs])^OB_SR[i]
            mm=iii==i
            if mm.sum():Rte[mm]=np.column_stack([labels(p,ob_full(Cte[mm],i)) for p in pufs])^OB_SR[i]
        accs=[]
        for j in range(4): accs.append(lr_acc(Cob,Rob[:,j],Cte,Rte[:,j],asd))
        rows.append(('OB-nz01',inst,asd,round(float(np.mean(accs)),4)))
    print('OB',inst,round(time.time()-t0,1),flush=True)
for inst in range(10):
    for asd in [0,1,2]:
        puf=ArbiterPUF(n=64,seed=200+inst,noisiness=0.1)
        rng=np.random.default_rng(8100+inst*10+asd)
        Ctr=rng.integers(0,2,size=(10000,68)).astype(np.uint8);Cte=rng.integers(0,2,size=(5000,68)).astype(np.uint8)
        a=lr_acc(Ctr,labels(puf,lfsr_int(Ctr)),Cte,labels(puf,lfsr_int(Cte)),asd)
        rows.append(('LFSR-nz01',inst,asd,round(a,4)))
    print('LFSR',inst,round(time.time()-t0,1),flush=True)
for inst in range(5):
    for asd in [0,1]:
        puf=ArbiterPUF(n=64,seed=100+inst,noisiness=0.1)
        K=np.random.default_rng(8200+inst).integers(0,2,size=(32,64)).astype(np.uint8);bb=np.random.default_rng(8201+inst).integers(0,2,size=(32,))
        rtr=np.random.default_rng(8202+inst*10+asd);rte=np.random.default_rng(8203+inst*10+asd)
        Ctr=rtr.integers(0,2,size=(200000,64)).astype(np.uint8);ii=rtr.integers(0,32,size=200000);jj=rtr.integers(0,32,size=200000)
        Cte=rte.integers(0,2,size=(5000,64)).astype(np.uint8);iii=rte.integers(0,32,size=5000);jjj=rte.integers(0,32,size=5000)
        ytr=(labels(puf,Ctr^K[ii])^bb[jj]).astype(int);yte=(labels(puf,Cte^K[iii])^bb[jjj]).astype(int)
        rows.append(('RSO-nz01-m32-200k',inst,asd,round(lr_acc(Ctr,ytr,Cte,yte,asd),4)))
    print('RSO-nz',inst,round(time.time()-t0,1),flush=True)
with open('experiments/encoding-induced-false-security-pufs-fullC-noisy.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['design','instance','seed','acc']);w.writerows(rows)
print('elapsed',round(time.time()-t0,1))
