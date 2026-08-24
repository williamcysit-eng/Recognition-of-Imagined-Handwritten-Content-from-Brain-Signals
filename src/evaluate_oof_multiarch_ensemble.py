import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:sys.path.append(ROOT)

from models import DeepConvNet,EEGNet82,GraphEEGNet
from src.extract import EEGDataset
from src.train_oof_aligned_dcn import align
from src.train_oof_dcn import make_oof_splits

NAMES=('dcn','aligned_dcn','eegnet_k25','eegnet_k15_swa','graph')


def predict(model,data,labels,device):
    out=[];model.eval()
    with torch.no_grad():
        for x,_ in DataLoader(EEGDataset(data,labels),batch_size=128):out.append(model(x.to(device)).cpu())
    return torch.cat(out)


def compositions(total,count,prefix=()):
    if count==1:
        yield prefix+(total,);return
    for value in range(total+1):yield from compositions(total-value,count-1,prefix+(value,))


def main():
    arc=np.load(os.path.join(ROOT,'data','processed','eeg_dataset.npz'));data=arc['data'].astype(np.float32);labels=arc['labels_0indexed']
    splits,test_idx=make_oof_splits(labels,5);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    oof=[[] for _ in NAMES];tests=[[] for _ in NAMES];targets=[]
    for fold,(_,val_idx) in enumerate(splits):
        dcn=DeepConvNet(24,26,501,temporal_kernel=15,dropout_rate=.5).to(device);dcn.load_state_dict(torch.load(os.path.join(ROOT,'models','checkpoints','oof_dcn_0_2000',f'fold_{fold}_seed_{42+fold}.pth'),map_location=device))
        aligned=DeepConvNet(24,26,501,temporal_kernel=15,dropout_rate=.5).to(device);aligned.load_state_dict(torch.load(os.path.join(ROOT,'models','checkpoints','oof_aligned_dcn',f'fold_{fold}_seed_{442+fold}.pth'),map_location=device))
        k25=EEGNet82(24,26,input_time_points=801,temporal_kernel_length=25,dropout_rate=.3).to(device);k25.load_state_dict(torch.load(os.path.join(ROOT,'models','checkpoints','oof_eegnet_k25',f'fold_{fold}_seed_{142+fold}.pth'),map_location=device))
        k15=EEGNet82(24,26,input_time_points=801,temporal_kernel_length=15,dropout_rate=.3).to(device);k15.load_state_dict(torch.load(os.path.join(ROOT,'models','checkpoints','oof_eegnet_k15_swa',f'fold_{fold}_seed_{342+fold}.pth'),map_location=device))
        graph=GraphEEGNet().to(device);graph.load_state_dict(torch.load(os.path.join(ROOT,'models','checkpoints','oof_graph',f'fold_{fold}_seed_{242+fold}.pth'),map_location=device))
        raw_val=data[val_idx,:,50:551];raw_test=data[test_idx,:,50:551]
        inputs=((dcn,raw_val,raw_test),(aligned,align(raw_val),align(raw_test)),(k25,data[val_idx],data[test_idx]),(k15,data[val_idx],data[test_idx]),(graph,raw_val,raw_test))
        targets.append(torch.from_numpy(labels[val_idx]).long())
        for index,(model,val_x,test_x) in enumerate(inputs):
            oof[index].append(predict(model,val_x,labels[val_idx],device));tests[index].append(predict(model,test_x,labels[test_idx],device))
        print(f'Loaded fold {fold+1}/5',flush=True)
    oof=[torch.cat(x) for x in oof];target=torch.cat(targets);scales=[oof[0].std()/x.std() for x in oof];oof=[x*s for x,s in zip(oof,scales)]
    best_accuracy,best_weights=0,None
    for weights in compositions(20,len(NAMES)):
        logits=sum(w*x for w,x in zip(weights,oof));acc=100*(logits.argmax(1)==target).float().mean().item()
        if acc>best_accuracy:best_accuracy,best_weights=acc,weights
    test_logits=[torch.stack(x).mean(0)*s for x,s in zip(tests,scales)];test_target=torch.from_numpy(labels[test_idx]).long()
    test_accuracy=100*(sum(w*x for w,x in zip(best_weights,test_logits)).argmax(1)==test_target).float().mean().item()
    print('Architectures:',', '.join(NAMES));print('OOF scales:',', '.join(f'{float(x):.6f}' for x in scales));print('OOF-selected integer weights:',best_weights)
    print(f'OOF accuracy: {best_accuracy:.2f}%');print(f'Held-out test accuracy: {test_accuracy:.2f}%')


if __name__=='__main__':main()
