import argparse,os,sys,time
import numpy as np,torch
import torch.nn as nn
from torch.utils.data import DataLoader
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.append(ROOT) if ROOT not in sys.path else None
from models import EEGNet82
from src.extract import EEGDataset
from src.run_utils import guard_output,prepare_run_dir
from src.train import load_and_split_data_pipeline,set_seed

def evaluate(m,l,d):
    m.eval();s=c=n=0
    with torch.no_grad():
        for x,y in l:x,y=x.to(d),y.to(d);z=m(x);s+=nn.functional.cross_entropy(z,y).item()*len(y);c+=(z.argmax(1)==y).sum().item();n+=len(y)
    return s/n,100*c/n
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id');p.add_argument('--runs-root',default=os.path.join(ROOT,'runs'));p.add_argument('--force',action='store_true');a=p.parse_args();run_dir=prepare_run_dir('experiment-center-loss',a.run_id,a.runs_root,a.force);print(f'Run directory: {run_dir}')
    set_seed(42);d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');tr,ty,va,vy,_,_,_,_=load_and_split_data_pipeline(os.path.join(ROOT,'data','processed','eeg_dataset.npz'));tl=DataLoader(EEGDataset(tr,ty),64,shuffle=True);vl=DataLoader(EEGDataset(va,vy),128)
    m=EEGNet82(24,26,input_time_points=801,temporal_kernel_length=15,dropout_rate=.3).to(d);centers=nn.Parameter(torch.randn(26,128,device=d)*.02);opt=torch.optim.AdamW(list(m.parameters())+[centers],lr=.003,weight_decay=.03);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=3,min_lr=1e-5);best=(1e9,0);stale=0;path=os.path.join(run_dir,'checkpoints','eegnet_center_loss_seed42.pth');guard_output(path,force=a.force)
    for e in range(1,90):
        t=time.time();m.train()
        for x,y in tl:
            x,y=x.to(d),y.to(d);x=x+torch.randn_like(x)*.04;emb=m.forward_embedding(x);class_logits=m.fc[4](m.fc[3](emb));prototype_logits=nn.functional.normalize(emb,dim=1)@nn.functional.normalize(centers,dim=1).T/.1;loss=nn.functional.cross_entropy(class_logits,y,label_smoothing=.1)+.2*nn.functional.cross_entropy(prototype_logits,y);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
        vl0,acc=evaluate(m,vl,d);sched.step(vl0);print(f'epoch={e:03d} val_loss={vl0:.4f} val_acc={acc:.2f}% time={time.time()-t:.1f}s',flush=True)
        if vl0<best[0]:best=(vl0,acc);stale=0;torch.save(m.state_dict(),path)
        else:stale+=1
        if stale>=20:break
    print('BEST CENTER LOSS',best)
if __name__=='__main__':main()
