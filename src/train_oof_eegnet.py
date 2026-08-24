import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models import EEGNet82
from src.extract import EEGDataset
from src.train import set_seed, train_deep_learning_model
from src.train_oof_dcn import make_oof_splits


def predict(model, x, y, device):
    logits, targets = [], []
    model.eval()
    with torch.no_grad():
        for bx, by in DataLoader(EEGDataset(x, y), batch_size=128):
            logits.append(model(bx.to(device)).cpu()); targets.append(by)
    return torch.cat(logits), torch.cat(targets)


def load_fold(path, device, kernel):
    model = EEGNet82(24, 26, input_time_points=801,
                     temporal_kernel_length=kernel, dropout_rate=0.3)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=142)
    parser.add_argument('--kernel', type=int, default=25)
    parser.add_argument('--swa', action='store_true')
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    archive=np.load(os.path.join(ROOT,'data','processed','eeg_dataset.npz'))
    data=archive['data'].astype(np.float32);labels=archive['labels_0indexed']
    splits,test_idx=make_oof_splits(labels,args.folds);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    suffix=f'oof_eegnet_k{args.kernel}' + ('_swa' if args.swa else '')
    outdir=os.path.join(ROOT,'models','checkpoints',suffix);os.makedirs(outdir,exist_ok=True)
    oof_correct=oof_total=0;test_logits=[]
    for fold,(train_idx,val_idx) in enumerate(splits):
        path=os.path.join(outdir,f'fold_{fold}_seed_{args.seed+fold}.pth')
        print(f"\n{'='*24} FOLD {fold+1}/{args.folds} {'='*24}",flush=True)
        if args.reuse and os.path.exists(path):model=load_fold(path,device,args.kernel)
        else:
            set_seed(args.seed+fold)
            model,_,device=train_deep_learning_model(
                'eegnet',data[train_idx],labels[train_idx],data[val_idx],labels[val_idx],24,801,
                num_epochs=args.epochs,batch_size=64,lr=.005,temporal_kernel=args.kernel,
                use_mixup=True,mixup_alpha=.2,noise_std=.07,use_swa=args.swa,swa_start_epoch=25,
                early_stopping_patience=20)
            torch.save(model.state_dict(),path)
        logits,targets=predict(model,data[val_idx],labels[val_idx],device)
        correct=(logits.argmax(1)==targets).sum().item();oof_correct+=correct;oof_total+=len(targets)
        print(f'fold_accuracy={100*correct/len(targets):.2f}%',flush=True)
        logits,_=predict(model,data[test_idx],labels[test_idx],device);test_logits.append(logits)
    targets=torch.from_numpy(labels[test_idx]).long();average=torch.stack(test_logits).mean(0)
    print(f'\nOOF accuracy: {100*oof_correct/oof_total:.2f}% ({oof_correct}/{oof_total})')
    print(f'Fold-ensemble test accuracy: {100*(average.argmax(1)==targets).float().mean().item():.2f}%')


if __name__=='__main__':main()
