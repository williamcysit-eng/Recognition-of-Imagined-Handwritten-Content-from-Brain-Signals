import argparse,os,sys,time
import numpy as np,torch
import torch.nn as nn
from torch.utils.data import DataLoader
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.append(ROOT) if ROOT not in sys.path else None
from models.masked_eeg_encoder import MaskedEEGAutoencoder,PretrainedEEGClassifier
from src.extract import EEGDataset
from src.run_utils import guard_output, prepare_run_dir
from src.train import load_and_split_data_pipeline,set_seed

def evaluate(m,l,d):
    m.eval();loss=nn.CrossEntropyLoss();s=c=n=0
    with torch.no_grad():
        for x,y in l:x,y=x.to(d),y.to(d);z=m(x);s+=loss(z,y).item()*len(y);c+=(z.argmax(1)==y).sum().item();n+=len(y)
    return s/n,100*c/n

def main():
    p=argparse.ArgumentParser();p.add_argument('--pretrain-epochs',type=int,default=25);p.add_argument('--finetune-epochs',type=int,default=80);p.add_argument('--seed',type=int,default=42);p.add_argument('--test',action='store_true');p.add_argument('--run-id');p.add_argument('--runs-root',default=os.path.join(ROOT,'runs'));p.add_argument('--force',action='store_true');a=p.parse_args();set_seed(a.seed);d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');run_dir=prepare_run_dir('experiment-masked-pretrain',a.run_id,a.runs_root,a.force);print(f'Run directory: {run_dir}')
    tr,ty,va,vy,te,tey,_,_=load_and_split_data_pipeline(os.path.join(ROOT,'data','processed','eeg_dataset.npz'));tr,va,te=(x[:,:,50:551].astype('float32') for x in (tr,va,te));mu=tr.mean((0,2),keepdims=True);sd=tr.std((0,2),keepdims=True)+1e-6;tr,va,te=((x-mu)/sd for x in (tr,va,te))
    tl=DataLoader(EEGDataset(tr,ty),64,shuffle=True);vl=DataLoader(EEGDataset(va,vy),128);tel=DataLoader(EEGDataset(te,tey),128)
    ae=MaskedEEGAutoencoder().to(d);opt=torch.optim.AdamW(ae.parameters(),lr=1e-3,weight_decay=.01)
    for epoch in range(1,a.pretrain_epochs+1):
        ae.train();total=0
        for x,_ in tl:
            x=x.to(d);masked=x.clone();mask=torch.zeros_like(x,dtype=torch.bool)
            channels=torch.rand(x.size(0),1,x.size(2),1,device=d)<.15;mask|=channels.expand_as(mask)
            starts=torch.randint(0,x.size(-1)-50,(x.size(0),),device=d);pos=torch.arange(x.size(-1),device=d).view(1,1,1,-1);tm=(pos>=starts.view(-1,1,1,1))&(pos<(starts+50).view(-1,1,1,1));mask|=tm.expand_as(mask);masked[mask]=0
            opt.zero_grad(set_to_none=True);out=ae(masked);loss=((out-x)[mask]**2).mean();loss.backward();opt.step();total+=loss.item()
        print(f'pretrain={epoch:03d} masked_mse={total/len(tl):.5f}',flush=True)
    m=PretrainedEEGClassifier().to(d);m.encoder.load_state_dict(ae.encoder.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=.02);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=3,min_lr=1e-5);ce=nn.CrossEntropyLoss(label_smoothing=.1);best=float('inf');stale=0;path=os.path.join(run_dir,'checkpoints',f'masked_pretrained_eeg_seed{a.seed}.pth');guard_output(path,force=a.force)
    for epoch in range(1,a.finetune_epochs+1):
        t=time.time();m.train()
        for x,y in tl:x,y=x.to(d),y.to(d);opt.zero_grad(set_to_none=True);loss=ce(m(x),y);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
        vl0,acc=evaluate(m,vl,d);sched.step(vl0)
        if vl0<best:best=vl0;stale=0;torch.save(m.state_dict(),path)
        else:stale+=1
        print(f'finetune={epoch:03d} val_loss={vl0:.4f} val_acc={acc:.2f}% time={time.time()-t:.1f}s',flush=True)
        if stale>=15:break
    m.load_state_dict(torch.load(path,map_location=d));print('BEST validation',evaluate(m,vl,d));print('FINAL test',evaluate(m,tel,d) if a.test else 'not evaluated')
if __name__=='__main__':main()
