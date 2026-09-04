"""Canonical checkpoint manifests for evaluation and tests."""

LEGACY_FIVE_MODEL_CHECKPOINTS = (
    # name, weight, filename, optional input time slice
    ("eegnet_k25_seed42", 3, "best_eegnet_k25_seed42.pth", None),
    ("eegnet_k15_swa_seed42", 1, "eegnet_k15_swa_seed42_standalone.pth", None),
    ("eegnet_k15_swa_seed123", 2, "eegnet_k15_swa_seed123.pth", None),
    ("dcn_window_0_2000", 2, "dcn_window_0_2000_seed42.pth", (50, 551)),
    ("graph_eeg_seed42", 2, "graph_eeg_seed42.pth", (50, 551)),
)

BEST_HYBRID_ARCHITECTURES = (
    "dcn",
    "aligned_dcn",
    "eegnet_k25",
    "eegnet_k15_swa",
    "graph",
)

BEST_HYBRID_WEIGHTS = (5, 3, 3, 9, 0)
BEST_HYBRID_SCALES = (1.0, 1.007047, 1.128674, 1.063061, 1.160222)

OOF_CHECKPOINTS_PER_FOLD = (
    ("oof_dcn_0_2000", "fold_{fold}_seed_{seed}.pth", (42, 542)),
    ("oof_aligned_dcn", "fold_{fold}_seed_{seed}.pth", (442,)),
    ("oof_eegnet_k25", "fold_{fold}_seed_{seed}.pth", (142,)),
    ("oof_eegnet_k15_swa", "fold_{fold}_seed_{seed}.pth", (342, 642)),
    ("oof_graph", "fold_{fold}_seed_{seed}.pth", (242,)),
)

FULL_REFIT_CHECKPOINTS = (
    "dcn_seed42.pth",
    "dcn_seed542.pth",
    "aligned_dcn_seed442.pth",
    "k25_seed142.pth",
    "k15_swa_seed342.pth",
    "k15_swa_seed642.pth",
)


def expected_oof_checkpoint_paths(folds=5):
    paths = []
    for directory, pattern, base_seeds in OOF_CHECKPOINTS_PER_FOLD:
        for fold in range(folds):
            for base_seed in base_seeds:
                paths.append(f"{directory}/{pattern.format(fold=fold, seed=base_seed + fold)}")
    return tuple(paths)


def full_refit_checkpoint_names(weights=BEST_HYBRID_WEIGHTS):
    names = []
    if weights[0] > 0:
        names.extend(("dcn_seed42.pth", "dcn_seed542.pth"))
    if weights[1] > 0:
        names.append("aligned_dcn_seed442.pth")
    if weights[2] > 0:
        names.append("k25_seed142.pth")
    if weights[3] > 0:
        names.extend(("k15_swa_seed342.pth", "k15_swa_seed642.pth"))
    if weights[4] > 0:
        names.append("graph_seed242.pth")
    return tuple(names)
