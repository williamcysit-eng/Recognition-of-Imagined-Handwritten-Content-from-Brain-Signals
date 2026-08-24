import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models import GraphEEGNet
from src.extract import EEGDataset
from src.train import set_seed
from src.train_oof_dcn import make_oof_splits


def evaluate(model, loader, device):
    model.eval(); loss_fn=nn.CrossEntropyLoss();loss_sum=correct=total=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device);z=model(x);loss_sum+=loss_fn(z,y).item()*len(y)
            correct+=(z.argmax(1)==y).sum().item();total+=len(y)
    return loss_sum/total,100*correct/total


def predict(model,data,labels,device):
    out=[];model.eval()
    with torch.no_grad():
        for x,_ in DataLoader(EEGDataset(data,labels),128):out.append(model(x.to(device)).cpu())
    return torch.cat(out)


def train_fold(data,labels,train_idx,val_idx,path,seed,epochs,device):
    set_seed(seed);model=GraphEEGNet().to(device)
    train=DataLoader(EEGDataset(data[train_idx],labels[train_idx]),64,shuffle=True)
    val=DataLoader(EEGDataset(data[val_idx],labels[val_idx]),128)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.03)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=3,min_lr=1e-5)
    loss_fn=nn.CrossEntropyLoss(label_smoothing=.1);best=float('inf');stale=0;tmp=path+'.training.pth'
    for epoch in range(1,epochs+1):
        model.train()
        for x,y in train:
            x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True);loss=loss_fn(model(x),y)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        vl,acc=evaluate(model,val,device);scheduler.step(vl)
        if vl<best:best=vl;stale=0;torch.save(model.state_dict(),tmp)
        else:stale+=1
        print(f'epoch={epoch:03d} val_loss={vl:.4f} val_acc={acc:.2f}%',flush=True)
        if stale>=15:break
    model.load_state_dict(torch.load(tmp,map_location=device));torch.save(model.state_dict(),path);os.remove(tmp)
    return model


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=60);p.add_argument('--seed',type=int,default=242)
    p.add_argument('--reuse',action='store_true');args=p.parse_args()
    arc=np.load(os.path.join(ROOT,'data','processed','eeg_dataset.npz'));data=arc['data'].astype('float32')[:,:,50:551];labels=arc['labels_0indexed']
    splits,test=make_oof_splits(labels,5);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outdir=os.path.join(ROOT,'models','checkpoints','oof_graph');os.makedirs(outdir,exist_ok=True)
    correct=total=0;tests=[]
    for fold,(tr,va) in enumerate(splits):
        print(f"\n{'='*24} FOLD {fold+1}/5 {'='*24}",flush=True);path=os.path.join(outdir,f'fold_{fold}_seed_{args.seed+fold}.pth')
        if args.reuse and os.path.exists(path):
            model=GraphEEGNet().to(device);model.load_state_dict(torch.load(path,map_location=device))
        else:model=train_fold(data,labels,tr,va,path,args.seed+fold,args.epochs,device)
        z=predict(model,data[va],labels[va],device);n=(z.argmax(1)==torch.tensor(labels[va])).sum().item();correct+=n;total+=len(va)
        print(f'fold_accuracy={100*n/len(va):.2f}%');tests.append(predict(model,data[test],labels[test],device))
    target=torch.tensor(labels[test]);print(f'\nOOF accuracy: {100*correct/total:.2f}% ({correct}/{total})')
    print(f'Fold-ensemble test accuracy: {100*(torch.stack(tests).mean(0).argmax(1)==target).float().mean().item():.2f}%')


if __name__=='__main__':main()
