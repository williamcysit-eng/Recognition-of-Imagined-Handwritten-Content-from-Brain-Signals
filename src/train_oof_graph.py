import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models import GraphEEGNet
from src.extract import EEGDataset
from src.run_utils import guard_output, prepare_run_dir, save_torch_state
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


def train_fold(data,labels,train_idx,val_idx,path,seed,epochs,device,force=False):
    set_seed(seed);model=GraphEEGNet().to(device)
    train=DataLoader(EEGDataset(data[train_idx],labels[train_idx]),64,shuffle=True)
    val=DataLoader(EEGDataset(data[val_idx],labels[val_idx]),128)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.03)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=3,min_lr=1e-5)
    loss_fn=nn.CrossEntropyLoss(label_smoothing=.1);best=float('inf');stale=0;best_state=None
    for epoch in range(1,epochs+1):
        model.train()
        for x,y in train:
            x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True);loss=loss_fn(model(x),y)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        vl,acc=evaluate(model,val,device);scheduler.step(vl)
        if vl<best:best=vl;stale=0;best_state=copy.deepcopy(model.state_dict())
        else:stale+=1
        print(f'epoch={epoch:03d} val_loss={vl:.4f} val_acc={acc:.2f}%',flush=True)
        if stale>=15:break
    if best_state is None:raise RuntimeError('Training produced no validation checkpoint')
    model.load_state_dict(best_state);save_torch_state(model,path,force=force)
    return model


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=60);p.add_argument('--seed',type=int,default=242)
    output=p.add_mutually_exclusive_group();output.add_argument('--reuse',action='store_true');output.add_argument('--force',action='store_true')
    p.add_argument('--cpu',action='store_true');p.add_argument('--data-path',default=os.path.join(ROOT,'data','processed','eeg_dataset.npz'))
    p.add_argument('--output-root');p.add_argument('--run-id');p.add_argument('--runs-root',default=os.path.join(ROOT,'runs'));args=p.parse_args()
    arc=np.load(args.data_path);data=arc['data'].astype('float32')[:,:,50:551];labels=arc['labels_0indexed']
    splits,test=make_oof_splits(labels,5);device=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    if args.output_root:checkpoint_root=args.output_root
    else:
        run_dir=prepare_run_dir('oof-graph',args.run_id,args.runs_root,args.force);checkpoint_root=os.path.join(run_dir,'checkpoints');print(f'Run directory: {run_dir}')
    outdir=os.path.join(checkpoint_root,'oof_graph');os.makedirs(outdir,exist_ok=True)
    correct=total=0;tests=[]
    for fold,(tr,va) in enumerate(splits):
        print(f"\n{'='*24} FOLD {fold+1}/5 {'='*24}",flush=True);path=os.path.join(outdir,f'fold_{fold}_seed_{args.seed+fold}.pth')
        if args.reuse and os.path.exists(path):
            model=GraphEEGNet().to(device);model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
        else:
            guard_output(path,force=args.force);model=train_fold(data,labels,tr,va,path,args.seed+fold,args.epochs,device,args.force)
        z=predict(model,data[va],labels[va],device);n=(z.argmax(1)==torch.tensor(labels[va])).sum().item();correct+=n;total+=len(va)
        print(f'fold_accuracy={100*n/len(va):.2f}%');tests.append(predict(model,data[test],labels[test],device))
    target=torch.tensor(labels[test]);print(f'\nOOF accuracy: {100*correct/total:.2f}% ({correct}/{total})')
    print(f'Fold-ensemble test accuracy: {100*(torch.stack(tests).mean(0).argmax(1)==target).float().mean().item():.2f}%')


if __name__=='__main__':main()
