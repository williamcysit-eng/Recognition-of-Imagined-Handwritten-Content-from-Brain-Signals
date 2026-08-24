import argparse
import os
import sys

import numpy as np
import torch

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:sys.path.append(ROOT)

from models import DeepConvNet
from src.train import set_seed,train_deep_learning_model
from src.train_oof_dcn import make_oof_splits,predict


def align(x):
    cov=np.einsum('nct,ndt->cd',x,x)/(len(x)*x.shape[2])
    cov+=np.eye(cov.shape[0])*1e-4*np.trace(cov)/len(cov)
    values,vectors=np.linalg.eigh(cov)
    inverse=(vectors*(1/np.sqrt(np.maximum(values,1e-8))))@vectors.T
    return np.einsum('cd,ndt->nct',inverse,x).astype(np.float32)


def load(path,device):
    m=DeepConvNet(24,26,501,temporal_kernel=15,dropout_rate=.5)
    m.load_state_dict(torch.load(path,map_location=device));return m.to(device)


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=80);p.add_argument('--seed',type=int,default=442);p.add_argument('--reuse',action='store_true');args=p.parse_args()
    arc=np.load(os.path.join(ROOT,'data','processed','eeg_dataset.npz'));data=arc['data'].astype('float32')[:,:,50:551];labels=arc['labels_0indexed']
    splits,test=make_oof_splits(labels,5);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');outdir=os.path.join(ROOT,'models','checkpoints','oof_aligned_dcn');os.makedirs(outdir,exist_ok=True)
    correct=total=0;tests=[]
    for fold,(tr,va) in enumerate(splits):
        print(f"\n{'='*24} FOLD {fold+1}/5 {'='*24}",flush=True);path=os.path.join(outdir,f'fold_{fold}_seed_{args.seed+fold}.pth')
        atr,ava,ate=align(data[tr]),align(data[va]),align(data[test])
        if args.reuse and os.path.exists(path):m=load(path,device)
        else:
            set_seed(args.seed+fold);m,_,device=train_deep_learning_model('deep_conv_net',atr,labels[tr],ava,labels[va],24,501,num_epochs=args.epochs,batch_size=64,lr=.005,temporal_kernel=15,use_mixup=False,noise_std=0,early_stopping_patience=20);torch.save(m.state_dict(),path)
        z,t=predict(m,ava,labels[va],device);n=(z.argmax(1)==t).sum().item();correct+=n;total+=len(t);print(f'fold_accuracy={100*n/len(t):.2f}%');z,_=predict(m,ate,labels[test],device);tests.append(z)
    target=torch.tensor(labels[test]);print(f'\nOOF accuracy: {100*correct/total:.2f}% ({correct}/{total})');print(f'Fold-ensemble test accuracy: {100*(torch.stack(tests).mean(0).argmax(1)==target).float().mean().item():.2f}%')


if __name__=='__main__':main()
