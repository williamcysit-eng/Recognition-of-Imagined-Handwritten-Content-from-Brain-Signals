import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.multi_window_eeg_net import MultiWindowEEGNet
from src.extract import EEGDataset
from src.train import load_and_split_data_pipeline, set_seed


def evaluate(model, loader, device):
    model.eval(); loss_fn = nn.CrossEntropyLoss(); loss_sum = correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device); logits = model(x)
            loss_sum += loss_fn(logits, y).item() * len(y)
            correct += (logits.argmax(1) == y).sum().item(); total += len(y)
    return loss_sum / total, 100 * correct / total


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42); parser.add_argument('--test', action='store_true')
    args = parser.parse_args(); set_seed(args.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr,ty,va,vy,te,tey,_,_=load_and_split_data_pipeline(os.path.join(ROOT,'data','processed','eeg_dataset.npz'))
    loaders=[DataLoader(EEGDataset(x,y),64 if i==0 else 128,shuffle=i==0)
             for i,(x,y) in enumerate(((tr,ty),(va,vy),(te,tey)))]
    model=MultiWindowEEGNet().to(device); print('parameters',sum(p.numel() for p in model.parameters()))
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.03)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=3,min_lr=1e-5)
    loss_fn=nn.CrossEntropyLoss(label_smoothing=.1); best=float('inf');stale=0
    path=os.path.join(ROOT,'models','checkpoints',f'multi_window_seed{args.seed}.pth')
    for epoch in range(1,args.epochs+1):
        t=time.time();model.train()
        for x,y in loaders[0]:
            x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True);loss=loss_fn(model(x),y)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        vl,va_acc=evaluate(model,loaders[1],device);sched.step(vl)
        if vl<best:best=vl;stale=0;torch.save(model.state_dict(),path)
        else:stale+=1
        print(f'epoch={epoch:03d} val_loss={vl:.4f} val_acc={va_acc:.2f}% time={time.time()-t:.1f}s',flush=True)
        if stale>=20:break
    model.load_state_dict(torch.load(path,map_location=device));print('BEST validation',evaluate(model,loaders[1],device))
    if args.test:print('FINAL test',evaluate(model,loaders[2],device))


if __name__=='__main__':main()
