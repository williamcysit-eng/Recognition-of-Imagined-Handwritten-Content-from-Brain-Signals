from .eegnet import EEGNet82, apply_max_norm_constraints
from .deep_conv_net import DeepConvNet
from .eeg_res_tcn import EEGResTCN
from .shallow_conv_net import ShallowConvNet
from .filter_bank_net import FilterBankNet
from .eeg_conformer import EEGConformer
from .graph_eeg_net import GraphEEGNet, DynamicGraphEEGNet, MultiScaleGraphEEGNet
from .multi_window_eeg_net import MultiWindowEEGNet
from .masked_eeg_encoder import EEGEncoder, MaskedEEGAutoencoder, PretrainedEEGClassifier
from .eeg_inception import EEGInception
