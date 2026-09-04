# Extended Experimental Report

## Recognition of Imagined Handwritten Content from EEG

This document records the complete reproduction and model-development study performed on this repository. It complements the original [`README.md`](README.md), which describes the initial architectures and originally reported results.

The same results are also maintained in machine-readable form in [`EXPERIMENT_RESULTS.csv`](EXPERIMENT_RESULTS.csv).

The study had four goals:

1. reproduce the repository in an isolated environment;
2. determine whether the reported performance was reproducible;
3. evaluate substantially different EEG architectures rather than only tuning the original CNNs;
4. construct a stronger ensemble without directly optimizing weights on test labels.

The best checkpoint-based result obtained in this study is:

| Metric | Accuracy |
|---|---:|
| Repeated-seed OOF accuracy | **19.13%** |
| Held-out test accuracy | **29.10%** |

The strongest prediction is a fixed 50/50 hybrid of a repeated-seed chronological OOF ensemble and models refitted on the complete 270-trial-per-class development set. Architectures and fusion weights are selected exclusively from out-of-fold predictions before refitting.

---

## 1. Dataset

The dataset contains single-trial EEG recorded from one participant while imagining handwriting one of the 26 English letters.

| Property | Value |
|---|---:|
| Trials | 7,800 |
| Classes | 26 |
| Trials per class | 300 |
| EEG channels | 24 |
| Samples per trial | 801 |
| Time range | -200 to 3,000 ms |
| Sampling frequency | 250 Hz |
| Chance accuracy | 3.85% |

The MATLAB tensor has shape `(24, 801, 7800)`. The extraction script transposes it to `(7800, 24, 801)` and converts labels from `1..26` to `0..25`.

The supplied data has already undergone artifact removal, baseline correction, and 0.1--45 Hz band-pass filtering. Inspection nevertheless showed that trials were not numerically standardized to identical variance. Trial standard deviations varied approximately from 6.2 to 15.0 across the 1st--99th percentiles. This motivated explicit normalization experiments, although they did not improve the strongest models.

Channel order:

```text
Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8
T7  T8  P7 P8 Fz Cz Pz M1 M2 AFz CPz POz
```

---

## 2. Reproduction Environment

| Component | Version / device |
|---|---|
| Architecture | ARM64 (`aarch64`) |
| Python | 3.12.3 |
| PyTorch | 2.12.0+cu130 |
| CUDA runtime | 13.0 |
| GPU | NVIDIA GB10 |
| NumPy | 2.4.0 |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.0 |

Activate the repository-local environment:

```bash
source .venv/bin/activate
```

---

## 3. Data Splitting and Evaluation Policy

### Primary chronological split

| Split | Trials per class | Total trials |
|---|---:|---:|
| Training | 240 | 6,240 |
| Validation | 30 | 780 |
| Test | 30 | 780 |

For every letter, the first 80% of trials are used for training, the next 10% for validation, and the final 10% for testing. This is stricter than random splitting because it exposes temporal distribution shift caused by fatigue, impedance changes, adaptation, and acquisition drift.

Models and ensemble weights were selected using validation predictions. Failed models were not promoted because of a favorable test result. Repeated experiments nevertheless create indirect familiarity with the fixed test set, so differences of one or two trials must be interpreted cautiously. A publication-quality follow-up should reserve a new untouched test set or use additional participants.

---

## 4. Reproduced Baseline

| Model | Test accuracy |
|---|---:|
| Logistic regression | 13.72% |
| DeepConvNet | 18.46% |
| EEGNet k=15 with SWA | 21.54% |
| DeepConvNet + EEGNet | 22.56% |
| DeepConvNet + EEGNet with nominal BN adaptation | 22.56% |
| EEGNet k=25 | 18.72% |
| Original three-model ensemble | **23.21%** |

The logistic-regression accuracy exactly matched the original report, supporting the correctness of the downloaded data and split. Deep-model performance was lower than the highest value in the original README, which is plausible because EEG optimization is sensitive to CUDA kernels, framework versions, initialization order, and random-number consumption.

### Batch-normalization adaptation

The original `adapt_bn_stats()` updated BatchNorm statistics and then restored the old values, so it behaved like no adaptation. True adaptation reduced validation accuracy from 20.90% to 19.87%. It was therefore excluded. With only 780 target trials, global target statistics can erase useful training-domain information without improving class-conditional alignment.

---

## 5. New Model Experiments

### 5.1 Residual Temporal Convolutional Network

File: [`models/eeg_res_tcn.py`](models/eeg_res_tcn.py)

The model uses four temporal kernels `(7, 15, 31, 63)`, residual depthwise temporal blocks, GroupNorm, and dilations `(1, 2, 4)`.

| Variant | Validation accuracy |
|---|---:|
| Global mean/max pooling | 3.97% |
| Sixteen retained temporal bins | 6.54% |

Global pooling assumes translation invariance. That is inappropriate for stimulus-locked ERP decoding: a response at 200 ms and the same response at 1,500 ms do not have the same physiological meaning. Retaining temporal bins helped but remained far below specialized EEG CNNs.

### 5.2 ShallowConvNet / FBCSP

File: [`models/shallow_conv_net.py`](models/shallow_conv_net.py)

This model uses temporal filtering, full spatial convolution, squaring, average pooling, and logarithmic compression.

| Result | Accuracy |
|---|---:|
| Best observed validation accuracy | approximately 9.10% |

The square--pool--log operations emphasize band power. Imagined handwriting in this dataset appears to depend more on precisely time-locked ERP structure than stationary oscillatory power.

### 5.3 Fixed Filter-Bank network

File: [`models/filter_bank_net.py`](models/filter_bank_net.py)

The signal was decomposed into 1--4, 4--8, 8--13, 13--30, and 30--45 Hz FIR bands, then fused with learned temporal blocks.

| Result | Accuracy |
|---|---:|
| Best validation accuracy | 6.54% |

The result agrees with ShallowConvNet: explicit stationary band power is not the dominant discriminative signal for these 26 classes.

### 5.4 EEG Conformer

Files: [`models/eeg_conformer.py`](models/eeg_conformer.py) and [`src/experiment_conformer.py`](src/experiment_conformer.py)

The model combines a convolutional EEG tokenizer and a three-layer Transformer encoder. CLS-token aggregation and full-token flattening were both tested.

| Variant | Validation accuracy |
|---|---:|
| CLS-token aggregation | approximately 3.85% |
| Full-token flattening | approximately 3.85% |

A diagnostic test reached 100% training accuracy on 104 balanced examples after 50 steps. The implementation was functional, but the approximately one-million-parameter attention model memorized rather than generalized. Strong local ERP priors were more useful than generic capacity.

### 5.5 GraphEEGNet

Files: [`models/graph_eeg_net.py`](models/graph_eeg_net.py) and [`src/experiment_graph_eeg.py`](src/experiment_graph_eeg.py)

GraphEEGNet represents the 24 electrodes as a graph built from approximate 10--20 scalp coordinates. Its normalized graph is:

```text
A_norm = D^(-1/2) (A + I) D^(-1/2)
```

A graph-temporal block performs gated neighbor propagation:

```text
H' = H + TemporalConv(H + sigmoid(g) * A_norm H)
```

| Property | Value |
|---|---:|
| Parameters | 192,570 |
| Input window | 0--2,000 ms |
| Best-checkpoint validation accuracy | 16.54% |
| Highest single-epoch validation accuracy | 18.08% |

Nearby electrodes observe correlated cortical fields, so graph propagation is physiologically motivated. GraphEEGNet was not the strongest individual model, but its errors were sufficiently different to improve the final ensemble.

---

## 6. Temporal-Window Experiments

| Input window | Parameters | Best-checkpoint validation accuracy |
|---|---:|---:|
| Full -200--3,000 ms | 278,246 | 14.87--15.64% depending on seed |
| 0--1,500 ms | 168,006 | 17.56% |
| 0--2,000 ms | 199,206 | **18.08%** |

Removing the prestimulus period and final second increased signal-to-noise ratio and reduced model size. Early activity contains visual encoding, orthographic processing, motor planning, and imagined movement. The 1,500--2,000 ms segment remained useful, while 2,000--3,000 ms appeared predominantly noisy.

---

## 7. Additional Training and Inference Experiments

### Multi-seed ensemble

Independent seed-42 and seed-123 models produced partially decorrelated errors. The first expanded ensemble reached 22.44% validation and 24.10% test accuracy.

### Time-shift test-time augmentation

Symmetric shifts of `±1`, `±2`, `±4`, and `±8` samples all reduced validation accuracy. The learned representations depend on stimulus-locked latency, so inference-only temporal shifts misalign ERP components.

### Logistic stacking

Five models produced 130 concatenated logits. A regularized multinomial meta-classifier selected by internal five-fold validation achieved 18.08% internal CV and 23.08% test accuracy. With only 780 meta-training samples and 26 classes, learned class-specific interactions overfit more than simple fixed weights.

### Temperature calibration

| Fusion method | Best validation accuracy |
|---|---:|
| Raw weighted logits | **23.33%** |
| Temperature-calibrated logits | 23.08% |
| Temperature-calibrated probabilities | 22.95% |

Calibration changed confidence but did not improve top-1 accuracy.

### EEGNet k=7

The 28 ms EEGNet achieved 16.28% best-checkpoint validation accuracy. Adding it produced 22.95% ensemble validation but only 23.21% test accuracy, so it was excluded.

### ERP prototype correlation

Class-average ERP templates were constructed from training trials after 4-sample temporal averaging. Raw cosine similarity, baseline-corrected similarity, diagonal-distance scoring, and per-trial/per-channel standardized cosine similarity were compared.

| Prototype method | Validation accuracy |
|---|---:|
| Raw cosine | 5.38% |
| Raw diagonal distance | 5.77% |
| Baseline-corrected cosine | 6.92% |
| Baseline-corrected diagonal distance | 7.18% |
| Trial/channel-standardized cosine | **10.13%** |

The selected prototype classifier reached 11.28% test accuracy. Fusion with the neural ensemble selected zero prototype weight on validation, so the ensemble remained 24.36%. Averaging reveals class ERP structure, but most useful template information was already represented by the CNNs.

### Euclidean covariance alignment

Training, validation, and test domains were independently whitened using unlabeled average channel covariance before training a 0--2,000 ms DeepConvNet.

| Metric | Accuracy |
|---|---:|
| Best-loss validation checkpoint | 17.05% |
| Highest validation epoch | 19.62% |
| Test accuracy | **20.51%** |

Alignment substantially improved standalone test performance but received zero weight when fused with the fixed ensemble. It corrected domain scale but did not add complementary decisions.

### Multi-window parallel EEGNet

Three branches modeled 0--300, 300--1,000, and 1,000--2,000 ms separately.

| Metric | Accuracy |
|---|---:|
| Validation | 16.15% |
| Test | 18.85% |

The model received zero validation-selected ensemble weight. Explicit cognitive-stage separation was useful but weaker than the single 0--2,000 ms window model.

### Dynamic GraphEEGNet

A trial-specific attention graph was blended with the fixed physical electrode graph.

| Metric | Accuracy |
|---|---:|
| Validation | 15.51% |
| Test | 17.18% |

The dynamic graph was weaker than the fixed graph and received zero ensemble weight. Functional connectivity estimated from one noisy trial was too unstable; the anatomical graph was a stronger regularizer.

### EEG-specific augmentation

The windowed DeepConvNet was retrained with 8% channel dropout, 60 ms temporal masking, 10% channel-wise amplitude jitter, and small Gaussian noise.

| Metric | Accuracy |
|---|---:|
| Highest validation epoch | 18.59% |
| Best-loss checkpoint validation | 16.67% |
| Test | 19.10% |

Augmentation delayed overfitting but weakened the final validation-loss checkpoint and added no ensemble gain.

### Confusion-pair second stage

Half of the validation split identified recurrent top-2 pairs: B--Q, A--S, B--G, B--R, and D--M. Pair-specific linear classifiers were trained from coarse 80 ms ERP-bin features. On the other half of validation, every nonzero correction set was no better than the unmodified ensemble, so zero pairs were selected and test accuracy stayed 24.36%.

### Masked EEG self-supervised pretraining

File: [`src/experiment_masked_pretrain.py`](src/experiment_masked_pretrain.py)

A convolutional encoder was pretrained exclusively on the labeled training split while ignoring labels. Fifteen percent of channels and one 200 ms temporal region were masked, and a decoder reconstructed the missing standardized EEG signal. Masked reconstruction MSE decreased from 0.96 to 0.73 over 25 epochs.

| Encoder initialization | Best validation accuracy |
|---|---:|
| Random initialization | 6.67% |
| Masked reconstruction pretraining | **9.87%** |

Self-supervision produced a clear +3.20 percentage-point transfer gain for the same encoder, proving that the reconstruction task learned reusable structure. Absolute performance remained far below EEGNet and DeepConvNet, so the model was not evaluated on the test set or added to the ensemble. Reconstruction favored signal morphology but did not learn sufficiently class-discriminative representations.

---

## 8. Five-Fold Out-of-Fold Experiment

File: [`src/train_oof_dcn.py`](src/train_oof_dcn.py)

The final 30 trials per class remained untouched. The first 270 trials per class were divided into five chronological blocks of 54 trials per class. Each fold trained on four blocks and validated on the fifth using a 0--2,000 ms DeepConvNet.

| Fold | Validation accuracy |
|---|---:|
| 1 | 14.10% |
| 2 | 18.30% |
| 3 | 17.45% |
| 4 | 13.46% |
| 5 | 14.60% |
| Combined OOF | **15.58% (1094/7020)** |

The five fold models averaged to **20.90%** on the fixed test set.

OOF did not beat the final ensemble, but the 13.46--18.30% spread exposed substantial temporal non-stationarity. A single chronological validation block has high uncertainty. Each fold model also saw less training data, and that loss outweighed variance reduction from averaging.

Reload the folds with:

```bash
python src/train_oof_dcn.py --folds 5 --epochs 100 --seed 42 --reuse \
  --output-root models/checkpoints
```

### Multi-architecture OOF extension

The same five chronological folds were used to train EEGNet k=25 models. No sample was scored by the corresponding fold model during its own training.

| OOF system | OOF accuracy | Test accuracy |
|---|---:|---:|
| Five windowed DeepConvNet folds | 15.58% | 20.90% |
| Five EEGNet k=25 folds | 13.35% | 21.41% |
| OOF-selected DCN + EEGNet k=25 fusion | 17.88% | 27.18% |
| Five EEGNet k=15 SWA folds | 15.33% | 23.46% |
| Five GraphEEGNet folds | 13.06% | 21.03% |
| Four-architecture OOF fusion | 18.69% | **27.56%** |
| Five aligned DeepConvNet folds | 14.87% | 23.85% |
| Final five-architecture OOF fusion | **18.72%** | **27.56%** |
| Nested regularized probability stacking | 18.08% | 26.54% |
| Repeated-seed OOF fusion | **19.13%** | **28.33%** |
| Full-development refit | -- | **28.46%** |
| Fixed 50/50 OOF + refit hybrid | -- | **29.10%** |

The final integer weights are `(5, 1, 5, 8, 1)` for windowed DCN, aligned DCN, EEGNet k=25, EEGNet k=15 SWA, and GraphEEGNet. Every architecture is scaled to the OOF logit standard deviation of the windowed DCN. Weights and scales are computed from OOF predictions, not test labels. EEGNet k=15 SWA receives the largest weight despite weaker standalone OOF accuracy because its errors complement both DCN variants. Graph and aligned DCN receive small nonzero weights that improve OOF accuracy but do not change the final test count beyond the four-architecture result.

### Repeated-seed bagging and full-development refit

A second independent seed was trained for every Window DCN and EEGNet k=15 SWA fold. Averaging seeds inside each fold increased OOF accuracy from 18.72% to 19.13% and test accuracy from 27.56% to 28.33%. OOF re-selection changed weights to `(5, 3, 3, 9, 0)`, automatically removing GraphEEGNet after seed variance was reduced.

A third EEGNet k=15 SWA seed was also evaluated. It reached 14.69% OOF / 23.85% test; averaging all three seeds produced 16.35% OOF versus 16.48% for the selected two-seed pair. OOF therefore excluded the third seed. Non-uniform seed weights improved each architecture in isolation but reduced the joint ensemble to 19.07% OOF, so equal averaging of the two retained seeds remained final.

The selected nonzero architectures were then retrained on all 270 development trials per class using fixed epoch counts and the OOF-selected weights. This full-development refit reached 28.46%. A predeclared 50/50 average of scale-normalized OOF-fold and full-refit logits reached **29.10%**. No blend coefficient was searched on test labels.

Additional negative results were retained:

- class-wise reliability weights reverted to global weights under nested OOF and stayed at 28.33%;
- fixed recency fold weights `(1,2,3,4,5)` reduced test accuracy to 27.95%;
- strict cumulative walk-forward DCN produced stage accuracies of 4.06%, 10.68%, 11.18%, and 14.60%, for 10.13% combined forward accuracy and 15.51% final-stage test accuracy; early stages lacked enough training data and did not support recency weighting;
- multi-scale fixed GraphEEGNet reached 16.92% validation / 18.33% test and received zero ensemble weight;
- latent teacher-student pretraining improved validation to 12.44% but remained below EEG-specific CNNs;
- prototype center loss reached only 14.36% validation.

Train or reload the EEGNet folds:

```bash
python src/train_oof_eegnet.py --epochs 80 --folds 5 --seed 142 \
  --output-root runs/manual-oof/checkpoints
python src/train_oof_eegnet.py --epochs 80 --folds 5 --seed 142 --reuse \
  --output-root models/checkpoints
```

These commands are useful for an individual architecture experiment. Use
`train_best_hybrid.py` for a complete new multi-architecture run so that both DCN
seeds, both retained k=15 seeds, all other folds, cache, and refits share one run.

Additional fold commands:

```bash
python src/train_oof_eegnet.py --epochs 80 --seed 342 --kernel 15 --swa \
  --output-root runs/manual-oof/checkpoints
python src/train_oof_graph.py --epochs 60 --seed 242 \
  --output-root runs/manual-oof/checkpoints
python src/train_oof_aligned_dcn.py --epochs 80 --seed 442 \
  --output-root runs/manual-oof/checkpoints
```

Evaluate the fixed leakage-free fusion:

```bash
python src/evaluate_oof_multiarch_ensemble.py \
  --checkpoint-root models/checkpoints \
  --cache-path runs/published-oof-eval/outputs/oof_multiarch_logits.npz
```

---

## 9. Ensemble Development

| Stage | Validation | Test |
|---|---:|---:|
| Reproduced original three-model ensemble | -- | 23.21% |
| Multi-seed five-model ensemble | 22.44% | 24.10% |
| Ensemble with 0--2,000 ms DCN | 22.82% | 24.23% |
| Ensemble with GraphEEGNet | **23.33%** | **24.36%** |
| Multi-architecture OOF ensemble | 18.72% OOF | 27.56% |
| Repeated-seed OOF ensemble | **19.13% OOF** | **28.33%** |
| Fixed OOF/full-refit hybrid | -- | **29.10%** |

### Final model weights

| Model | Weight | Inductive bias |
|---|---:|---|
| EEGNet k=25, seed 42 | 3 | 100 ms filters and depthwise spatial processing |
| EEGNet k=15 SWA, seed 42 | 1 | Fast 60 ms dynamics and flatter SWA solution |
| EEGNet k=15 SWA, seed 123 | 2 | Independent stochastic trajectory |
| DeepConvNet, 0--2,000 ms | 2 | Hierarchical ERP features in a denoised window |
| GraphEEGNet, seed 42 | 2 | Explicit electrode-topology propagation |

```python
logits = (
    3 * eegnet_k25(x)
    + eegnet_k15_seed42(x)
    + 2 * eegnet_k15_seed123(x)
    + 2 * dcn_window(x[..., 50:551])
    + 2 * graph_eeg(x[..., 50:551])
)
prediction = logits.argmax(dim=1)
```

The gain comes from error decorrelation across temporal scales, seeds, input windows, and spatial priors rather than from one dominant model.

---

## 10. Current Reproduction

This report and `EXPERIMENT_RESULTS.csv` are the authoritative descriptions of
retained results. The root README is an operational guide; earlier single-split
results in its historical section are not the current final method.

### Train the best OOF/full-refit hybrid in one command

After preparing `data/processed/eeg_dataset.npz`:

```bash
source .venv/bin/activate
python src/train_best_hybrid.py --run-id best-hybrid-v1
```

The command performs the complete dependency chain:

1. train 35 class-wise ordered OOF checkpoints across five architecture groups;
2. calculate OOF logits and select scales and integer fusion weights;
3. store those selected values in the OOF cache;
4. refit the selected architecture families on all 270 development trials per class;
5. evaluate the OOF ensemble, full-development refit, and fixed 50/50 hybrid.

The retained checkpoints produce:

```text
Architectures: dcn, aligned_dcn, eegnet_k25, eegnet_k15_swa, graph
OOF-selected integer weights: (5, 3, 3, 9, 0)
OOF accuracy: 19.13%
Held-out test accuracy: 28.33%
Full-development refit: 28.46%
Fixed 50/50 hybrid: 29.10%
```

Every invocation writes to `runs/<run-id>/`, including configuration, metrics,
OOF cache, all checkpoints, and a SHA256 checkpoint inventory. An existing run is
rejected by default. `--force` is required to replace any output in that run.

The smoke-test path is isolated and therefore cannot damage retained checkpoints:

```bash
python src/train_best_hybrid.py --run-id smoke-001 --quick 1
```

Complete retraining reproduces the method, not necessarily bit-identical weights
or the exact 29.10% point estimate. CUDA kernels, stochastic optimization, and
early-stopping trajectories can change the selected OOF weights and final result.

### Evaluate retained best-hybrid checkpoints

When the ignored local dataset, OOF cache, and full-refit checkpoints are present:

```bash
python src/evaluate_hybrid_refit_oof.py
```

The evaluator reads scales and weights from a newly generated cache. Legacy caches
without those fields fall back to the retained `(5,3,3,9,0)` configuration.

### Historical five-model single-split workflow

`train_final_ensemble.py` is retained only for the earlier 24.36% fixed five-model
experiment. It now also writes to an isolated run directory:

```bash
python src/train_final_ensemble.py --run-id legacy-five-v1
python src/evaluate_deep_ensemble.py \
  --checkpoint-dir runs/legacy-five-v1/checkpoints
```

The retained historical checkpoints reproduce 23.33% validation and 24.36% test
accuracy. They are not the 29.10% best-hybrid pipeline.

### Regression tests

```bash
python -m unittest discover -s tests -v
```

The tests cover split isolation and balance, selected-model tensor shapes,
checkpoint manifests, default non-overwrite behavior, and fixed checkpoint
evaluation when the ignored local binary assets are installed.

After at least five predeclared complete runs, aggregate them with:

```bash
python src/summarize_runs.py runs/run-1/metrics.json runs/run-2/metrics.json \
  runs/run-3/metrics.json runs/run-4/metrics.json runs/run-5/metrics.json
```

The summary includes all individual values, mean, sample standard deviation, and
a 95% Student-t confidence interval.

---

## 11. Main Scientific Findings

1. **Time localization matters.** Models retaining explicit ERP time bins outperformed global-pooling networks.
2. **ERP structure matters more than stationary band power.** ShallowConvNet and the fixed filter bank were weak.
3. **The final second is mostly detrimental.** Cropping to 0--2,000 ms improved DeepConvNet.
4. **Electrode topology is useful as an ensemble prior.** GraphEEGNet improved the ensemble despite lower standalone accuracy.
5. **More capacity is not automatically better.** The Conformer memorized but failed to generalize.
6. **Temporal non-stationarity is a major limitation.** OOF folds differed by almost five percentage points.

---

## 12. Limitations

1. The dataset contains only one participant.
2. Trials are temporally ordered and class-blocked, so drift can interact with class identity.
3. Validation and test contain only 780 trials each; one trial changes accuracy by about 0.128 percentage points.
5. Graph coordinates are approximate rather than participant-specific digitized positions.
6. The 29.10% result is checkpoint-reproducible, but a full multi-run retraining distribution has not yet been measured.
7. The 29.10% hybrid corresponds to 227/780 correct trials, only five more than
   the 28.46% full-refit system. This small paired difference is not evidence of a
   statistically reliable improvement by itself.
8. The current dataset has no separately distributed acquisition timestamp. The
   protocol therefore relies on within-class sample order and should be described
   as a class-wise chronological split rather than a fully reconstructed global timeline.

---

## 13. Recommended Future Work

1. Freeze the current 780-trial test split for retrospective reporting only. Do
   not use it for further architecture, seed, epoch, calibration, or fusion choices.
2. Designate a newly collected or completely untouched external test set before
   the next development cycle; preferably include additional participants and
   retain participant/session identifiers for grouped evaluation.
3. Repeat the complete OOF/refit pipeline with at least five predeclared run seeds.
   Report the number of runs, mean, sample standard deviation, 95% confidence
   interval, and all individual run values—not only the best checkpoint.
4. Use digitized participant-specific electrode coordinates.
5. Add session/domain-adversarial objectives to reduce chronological drift.
6. For multi-participant data, report both within-participant and
   leave-one-participant-out performance with participant-level uncertainty.

The central lesson is that stronger performance came from combining appropriate EEG priors—short temporal filters, explicit ERP windows, multiple stochastic solutions, and electrode topology—rather than simply increasing depth or parameter count.
