import argparse,os,sys
import numpy as np,torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.append(ROOT) if ROOT not in sys.path else None
from models import DeepConvNet
from src.train import set_seed,train_deep_learning_model
from src.train_oof_dcn import make_oof_splits,predict
from src.run_utils import guard_output,prepare_run_dir,save_torch_state

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id');p.add_argument('--runs-root',default=os.path.join(ROOT,'runs'));p.add_argument('--force',action='store_true');p.add_argument('--oof-checkpoint-root',default=os.path.join(ROOT,'models','checkpoints'));args=p.parse_args();run_dir=prepare_run_dir('experiment-walk-forward',args.run_id,args.runs_root,args.force);print(f'Run directory: {run_dir}')
    a=np.load(os.path.join(ROOT,'data','processed','eeg_dataset.npz'));x=a['data'].astype('float32')[:,:,50:551];y=a['labels_0indexed'];splits,test=make_oof_splits(y,5);blocks=[v for _,v in splits];d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');outdir=os.path.join(run_dir,'checkpoints','walk_forward_dcn');os.makedirs(outdir,exist_ok=True);correct=total=0;last=None
    for stage in range(1,5):
        train=np.concatenate(blocks[:stage]);val=blocks[stage];path=os.path.join(outdir,f'stage_{stage}_seed_{742+stage}.pth');print(f'\nSTAGE {stage}: train_blocks=0..{stage-1} validate_block={stage}',flush=True)
        if stage==4:
            path=os.path.join(args.oof_checkpoint_root,'oof_dcn_0_2000','fold_4_seed_46.pth');m=DeepConvNet(24,26,501,temporal_kernel=15,dropout_rate=.5).to(d);m.load_state_dict(torch.load(path,map_location=d,weights_only=True))
        else:
            guard_output(path,force=args.force);set_seed(742+stage);m,_,d=train_deep_learning_model('deep_conv_net',x[train],y[train],x[val],y[val],24,501,num_epochs=70,batch_size=64,lr=.005,temporal_kernel=15,use_mixup=False,noise_std=0,early_stopping_patience=15);save_torch_state(m,path,force=args.force)
        z,t=predict(m,x[val],y[val],d);n=(z.argmax(1)==t).sum().item();correct+=n;total+=len(t);print(f'forward_accuracy={100*n/len(t):.2f}%');last=m
    z,t=predict(last,x[test],y[test],d);print(f'\nCUMULATIVE_FORWARD_ACCURACY={100*correct/total:.2f}% ({correct}/{total})');print(f'LAST_STAGE_TEST={100*(z.argmax(1)==t).float().mean().item():.2f}%')
if __name__=='__main__':main()
