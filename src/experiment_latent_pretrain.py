import argparse,os,sys
import numpy as np,torch
import torch.nn as nn
from torch.utils.data import DataLoader
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.append(ROOT) if ROOT not in sys.path else None
from models import MaskedEEGAutoencoder,EEGEncoder,PretrainedEEGClassifier
from src.extract import EEGDataset
from src.run_utils import prepare_run_dir,write_json
from src.train import load_and_split_data_pipeline,set_seed

def mask_input(x):
    masked=x.clone();mask=torch.rand(x.size(0),1,x.size(2),1,device=x.device)<.15;starts=torch.randint(0,x.size(-1)-50,(x.size(0),),device=x.device);p=torch.arange(x.size(-1),device=x.device).view(1,1,1,-1);mask=mask.expand_as(x)|(((p>=starts[:,None,None,None])&(p<(starts+50)[:,None,None,None])).expand_as(x));masked[mask]=0;return masked,mask
def evaluate(m,l,d):
    m.eval();s=c=n=0
    with torch.no_grad():
        for x,y in l:x,y=x.to(d),y.to(d);z=m(x);s+=nn.functional.cross_entropy(z,y).item()*len(y);c+=(z.argmax(1)==y).sum().item();n+=len(y)
    return s/n,100*c/n
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id');p.add_argument('--runs-root',default=os.path.join(ROOT,'runs'));p.add_argument('--force',action='store_true');a=p.parse_args();run_dir=prepare_run_dir('experiment-latent-pretrain',a.run_id,a.runs_root,a.force);print(f'Run directory: {run_dir}')
    set_seed(42);d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');tr,ty,va,vy,_,_,_,_=load_and_split_data_pipeline(os.path.join(ROOT,'data','processed','eeg_dataset.npz'));tr,va=(x[:,:,50:551].astype('float32') for x in (tr,va));mu=tr.mean((0,2),keepdims=True);sd=tr.std((0,2),keepdims=True)+1e-6;tr=(tr-mu)/sd;va=(va-mu)/sd;tl=DataLoader(EEGDataset(tr,ty),64,shuffle=True);vl=DataLoader(EEGDataset(va,vy),128)
    ae=MaskedEEGAutoencoder().to(d);opt=torch.optim.AdamW(ae.parameters(),lr=1e-3,weight_decay=.01)
    for e in range(1,16):
        ae.train();total=0
        for x,_ in tl:x=x.to(d);mx,mask=mask_input(x);opt.zero_grad(set_to_none=True);out=ae(mx);loss=((out-x)[mask]**2).mean();loss.backward();opt.step();total+=loss.item()
        print(f'reconstruction={e:02d} mse={total/len(tl):.5f}',flush=True)
    teacher=ae.encoder.eval();[p.requires_grad_(False) for p in teacher.parameters()];student=EEGEncoder().to(d);student.load_state_dict(teacher.state_dict());predictor=nn.Conv1d(128,128,1).to(d);opt=torch.optim.AdamW(list(student.parameters())+list(predictor.parameters()),lr=5e-4,weight_decay=.01)
    for e in range(1,16):
        student.train();total=0
        for x,_ in tl:
            x=x.to(d);mx,_=mask_input(x)
            with torch.no_grad():target=nn.functional.normalize(teacher(x),dim=1)
            pred=nn.functional.normalize(predictor(student(mx)),dim=1);loss=(1-(pred*target).sum(1)).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=loss.item()
        print(f'latent={e:02d} cosine_loss={total/len(tl):.5f}',flush=True)
    model=PretrainedEEGClassifier().to(d);model.encoder.load_state_dict(student.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.02);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=3,min_lr=1e-5);ce=nn.CrossEntropyLoss(label_smoothing=.1);best=(1e9,0);stale=0
    for e in range(1,50):
        model.train()
        for x,y in tl:x,y=x.to(d),y.to(d);opt.zero_grad(set_to_none=True);loss=ce(model(x),y);loss.backward();opt.step()
        vl0,acc=evaluate(model,vl,d);sched.step(vl0);print(f'finetune={e:02d} val_loss={vl0:.4f} val_acc={acc:.2f}%',flush=True)
        if vl0<best[0]:best=(vl0,acc);stale=0
        else:stale+=1
        if stale>=15:break
    print('BEST LATENT PRETRAIN',best);write_json(os.path.join(run_dir,'metrics.json'),{'best_validation_loss':best[0],'best_validation_accuracy':best[1]},force=a.force)
if __name__=='__main__':main()
