import os
import numpy as np
from scipy.special import log_softmax

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOBAL=np.asarray([5,3,3,9,0],dtype=np.float64)


def class_weights(logits,targets,tau):
    losses=np.zeros((logits.shape[1],26),dtype=np.float64)
    lp=log_softmax(logits,axis=2)
    for model in range(logits.shape[1]):
        for cls in range(26):
            idx=targets==cls;losses[model,cls]=-lp[idx,model,cls].mean()
    if np.isinf(tau):return np.repeat(GLOBAL[:,None],26,axis=1)
    weights=GLOBAL[:,None]*np.exp(-losses/tau)
    return weights*(GLOBAL.sum()/(weights.sum(0,keepdims=True)+1e-12))


def predict(logits,weights):
    return (logits*weights.T[None,:,:].transpose(0,2,1)).sum(1).argmax(1)


def main():
    d=np.load(os.path.join(ROOT,'outputs','oof_multiarch_logits.npz'));oof=d['oof'];test=d['test'];y=d['oof_targets'];ty=d['test_targets'];fold=d['fold_ids']
    scales=oof[:,0].std(axis=(0,1))/oof.std(axis=(0,2));oof=oof*scales[None,:,None];test=test*scales[None,:,None]
    best=(0,None)
    for tau in (.1,.2,.5,1.,2.,5.,10.,np.inf):
        pred=np.empty_like(y)
        for held in range(5):
            tr=fold!=held;va=~tr;pred[va]=predict(oof[va],class_weights(oof[tr],y[tr],tau))
        acc=100*(pred==y).mean();print(f'tau={tau} nested_oof={acc:.2f}%')
        if acc>best[0]:best=(acc,tau)
    weights=class_weights(oof,y,best[1]);test_acc=100*(predict(test,weights)==ty).mean()
    print(f'SELECTED tau={best[1]} OOF={best[0]:.2f}%');print(f'HELD_OUT_TEST={test_acc:.2f}%')


if __name__=='__main__':main()
