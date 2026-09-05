# Suggested improvements

## 1. Conclusion and accepted baseline

**Yes: meaningful architectural improvements remain untested.** The most promising next steps are a compact spectral-power model, learned long-range temporal interactions, and a smaller classifier head that preserves temporal position. These are more substantial changes in what the model can learn than another optimizer, attention module, or nearby kernel-size sweep.

Use only these recent results as the authoritative baseline:

| Model | Test accuracy |
|---|---:|
| DeepConvNet | 19.49% |
| EEGNet (k=15, SWA) | 17.44% |
| EEGNet (k=25) | 16.67% |
| 2-model ensemble | 21.79% |
| 3-model ensemble | 22.18% |

The dataset contains one participant and 7,800 total trials. Under the documented 80/10/10 split, fitting uses 6,240 trials: **240 per class**, not 300. The remaining trials provide 780 validation and 780 test examples.

The older numbers in `process.md` and the remaining historical tables in `README.md` are **not evidence of achievable accuracy or of a performance ceiling**. I use that history only to identify attempted techniques. In particular, explanations such as “SWA destroys diversity,” “DCN is at a delicate optimum,” and “contrastive learning cannot transfer” are hypotheses, not established mechanisms.

On the documented 780-trial test set, the rounded ensemble scores correspond to 170 versus 173 correct predictions. The third model therefore adds **three net correct predictions, or approximately 0.39 percentage points**. This is a valid measured improvement, but not enough by itself to establish a reliable advantage. An illustrative independent-trial Wilson 95% interval for 173/780 is approximately 19.4–25.2%; temporal dependence and adaptive model selection are not accounted for by that interval. Comparing two models requires paired predictions, not overlapping or non-overlapping individual intervals.

**Recommended order:**

| Priority | Experiment | Why it deserves attention |
|---|---|---|
| Prerequisite | Validation-only experimentation, independent model RNGs, explicit checkpoint/SWA handling | Make comparisons interpretable without repeatedly consulting the test set |
| 1 | FBCNet-inspired filter-bank/log-variance model, initially as a separate ensemble member | Introduce an explicit spectral-power representation rather than another similar waveform CNN |
| 2 | Small residual dilated temporal-convolution network | Learn nonlinear relationships between events separated by hundreds of milliseconds or seconds |
| 3 | Position-preserving classifier-head compression | Reduce parameter concentration without discarding the temporal alignment that may matter here |
| 4 | Early/late-window specialization | Test whether different portions of the cue-and-imagery epoch need different representations |
| Later | Small convolutional transformer | Investigate global interactions only after establishing a strong, smaller temporal-convolution comparator |

All prospective benefits below are **[INFERENCE] / hypotheses to test on development data**, not promised accuracy gains. No model training, checkpoint evaluation, or new test-set analysis was performed for this document.

## 2. Establish an honest comparison protocol first

### 2.1 Freeze the test set; do not optimize against it

The current entry point evaluates test accuracy after training individual models and several ensemble variants, then runs a test-scored logistic-regression baseline (`src/train.py:575–693`). That makes ordinary experimentation expose the test result repeatedly.

Before new experiments, provide a **development-only execution path** that never evaluates the test set. This is a proposed change, not an existing CLI option. Choose architectures, training duration, SWA policy, normalization, windows, and ensemble weights using development data only. Lock the complete prediction procedure before final testing.

Distinguish two issues:

- The supplied percentages are accepted as accurate measurements.
- If previous design decisions were made after inspecting test results, that set has already influenced development. Freezing it now prevents further adaptation but cannot retrospectively make it an untouched confirmatory set. Keep it fixed for historical comparison; obtain genuinely unseen recordings for a clean final confirmation when possible. Do not create a supposedly fresh test set by reshuffling trials already used for training or selection.

### 2.2 Verify acquisition groups, not just class balance

`load_and_split_data_pipeline` takes the first 80%, next 10%, and last 10% of **array indices within each class** (`src/train.py:96–116`). This implements the documented split, but the function has no acquisition timestamps or session IDs with which to prove global chronological separation.

Inspection of the MAT/NPZ schemas found no acquisition timestamps, session/run/block IDs, or original trial IDs. The available `time_points` vector is a shared cue-relative axis, not a sequence of acquisition times. `src/extract.py` preserves stored order; the provider instructions do not authenticate it as chronological order. Recovering these fields requires acquisition/export provenance, not a different splitter operating on the same archive.

For development comparisons:

1. Preserve the existing test membership.
2. Recover session/run/block IDs and original acquisition order where available.
3. Within the remaining 90%, use a small set of forward-chaining, group-preserving validation folds. Keep whole acquisition groups together, even if exact class balance becomes imperfect.
4. Separate early-stopping data from the outer fold used to score an experiment when using nested evaluation. Reusing one small validation set for many architecture decisions also overfits validation.
5. Keep every crop, filtered view, augmentation, and copy of a trial in that trial's partition.

If only array order is available, describe validation as an **index-block approximation**, not verified session generalization. Random stratified folds would not solve that limitation.

The documented epochs span −0.2 to +3.0 seconds while cues occur every 3 seconds. **[INFERENCE, conditional on that timing being literal]** adjacent epochs can contain overlapping continuous-recording samples. Use event timestamps to group overlapping epochs or purge boundary trials. The exclusion interval must also account for filter support; a fixed one-trial gap is not a universal guarantee. Provider preprocessing may already have mixed information across boundaries, so its fitting scope matters too.

There is also an important preprocessing uncertainty. The provider's [`data_instruction.docx`](data/raw/data_instruction.docx), Data section, documents 0.1–45 Hz filtering, baselining, and removal of some principal components; it warns that some trials may be rank-deficient. It does **not** substantiate README claims of ICA or per-trial z-scoring. The extractor only converts formats and labels. Obtain the upstream preprocessing settings and fitting scope before certifying independence; neither a new architecture nor new filtering can undo already-removed components or prove that upstream processing was split-safe.

### 2.3 Keep inference strictly inductive

For the objective requested here, a prediction should be a fixed function of one trial and frozen model state.

- Fit scalers, feature selectors, CSP filters, covariance references, learned normalization, and any preprocessing parameters on the fitting partition only.
- Do not update BatchNorm, perform entropy minimization, pretrain, or estimate an alignment transform using the test cohort, even without labels. Those are transductive procedures, not the strict held-out protocol proposed here.
- A fixed filter, STFT, crop, or predetermined per-trial normalization applied independently to one complete trial is different: it need not learn anything from other test trials. It is suitable for offline whole-trial decoding if used identically during development. Online prediction would require causal preprocessing and its own latency evaluation.
- No class-quota assignment, exploitation of class-sorted test order, trial-index features, test-label-informed trial averaging, or language-model constraints for this isolated-letter task.
- Verify that predictions do not depend materially on test batch order or companion trials, and that evaluation leaves model buffers unchanged.

## 3. Current implementation issues that affect the next experiments

These observations describe the current source, not a reconstruction or rejection of the accepted run.

### 3.1 The current BN adaptation path does not retain adapted statistics

`adapt_bn_stats` saves each BatchNorm running mean/variance, processes the supplied loader, and then restores those values before prediction (`src/train.py:392–406`). It does not restore `num_batches_tracked`.

A synthetic check of this exact function found:

- Running mean and variance restored.
- Evaluation outputs unchanged: maximum absolute difference **0.0**.
- Batch counter changed from 0 to 2.

No EEG or held-out data was used for this check. Consequently, README explanations attributing gains to retained test-time BN adaptation are not supported by this implementation. **Do not “fix” it by retaining test statistics to raise accuracy. Remove that test-data pass from the proposed strict evaluation path.** If normalization is improved, use training-only statistics instead.

### 3.2 Reassess SWA using the actual averaged model on validation

The current SWA implementation averages the entire `state_dict`, including BatchNorm buffers and counters, from a fixed starting epoch. It does not subsequently recalibrate BatchNorm on clean training inputs. Validation history and early stopping describe the live, non-averaged model, while the returned model can be the SWA model (`src/train.py:277–368`).

A worthwhile controlled comparison is:

1. Best validation checkpoint, without SWA.
2. Parameter-averaged model using PyTorch's `AveragedModel`, with BatchNorm recalculated on **clean fitting-partition inputs only**.
3. Evaluate both resulting prediction models on the same development folds, individually and in the ensemble.

Buffer averaging is itself a supported PyTorch option, so its presence alone does not prove that the current SWA result is wrong. The hypothesis is that a cleaner averaging/normalization policy and validation of the actual returned model could work better. Do not claim the 17.44% result proves either that SWA helps or that it hurts: the accepted baseline does not contain a matched k=15 non-SWA comparison.

For a custom BN refresh, reset BN statistics, run a deterministic clean training-data pass with only BN updating, disable Mixup/noise and dropout, retain the resulting training statistics, and restore evaluation mode. PyTorch's `update_bn` is a useful reference, but it calls `model.train()`; account explicitly for dropout rather than assuming it remains disabled. Select the refresh policy on development data.

### 3.3 Isolate model randomness and save what was actually evaluated

- `set_seed` initializes shared NumPy/PyTorch state once; subsequent model training consumes it (`src/train.py:41–48, 648–681`). Changing DCN training can therefore change the later EEGNet trajectory. CPU ensemble training also uses threads sharing process-global RNGs. Reset seeds before each sequential model run and use dedicated loader/augmentation generators; independent processes are safer than shared threads for concurrent reproducible runs.
- The CPU ensemble branch reloads `best_eegnet.pth` after receiving the returned model, discarding the returned SWA weights (`src/train.py:625–646`). Save and reload the explicitly selected checkpoint type instead.
- Both EEGNet kernels save to `best_eegnet.pth`, and the averaged model is not separately saved by the training function (`src/train.py:161–169, 306–322, 365–368`). Use run/model/kernel/seed-specific artifacts containing the actual evaluated state. This does not mean the in-memory results are invalid; it means later checkpoint reuse can represent a different model.
- Standalone commands are not necessarily the ensemble recipes: standalone EEGNet gets an approximately half-sampling-rate kernel (125 at 250 Hz), no SWA, and default zero noise; standalone DCN gets the generic Mixup flag (`src/train.py:556–584`). Make configuration explicit before comparing architecture names.
- Head changes must preserve an intentional constraint policy. `apply_max_norm_constraints` currently targets EEGNet's `fc.1.weight`, the hidden layer, not its `fc.4` output layer (`models/eegnet.py:175–180, 208–224`). Replacing or reordering the head can silently move the max-norm constraint onto a different layer. Treat any constraint change as explicit, not an accidental consequence of new module indices.

These changes primarily improve experiment validity and reproducibility. They are not claimed accuracy improvements.

## 4. Architectural improvements worth trying

### 4.1 First choice: a compact spectral-power branch

**Hypothesis:** explicit band-power features could supply information that the present waveform CNNs learn inefficiently, or supply useful ensemble complementarity.

Try an FBCNet-inspired model [1], or the related ShallowConvNet power-estimation structure [2]:

```text
One EEG trial
  -> fixed filter bank
  -> a few learned spatial filters per band
  -> log-variance over a few ordered time windows
  -> small regularized 26-class linear head
```

A deliberately small starting specification:

- Four bands: 4–8, 8–13, 13–30, and 30–45 Hz.
- Eight spatial filters per band.
- Four ordered temporal windows.
- Stable log-variance with an epsilon/clamp.
- Retain the existing broadband model alongside it, rather than discarding slow cue-related information below 4 Hz.

This gives 4 × 8 × 4 = **128 features** and a 3,354-parameter linear classifier, excluding spatial filters and normalization. These are starting design choices, not established physiological optima. Handle the 801-sample input explicitly when making windows; do not copy an implementation that assumes divisibility by four without checking its boundaries.

**Why this differs from previous attempts:** the existing EEGInception uses parallel temporal convolutions, not explicit band decomposition followed by log-variance. A failed Riemannian tangent-space experiment does not test this representation or its learned spatial filters. Conversely, ordinary CNNs can already learn power-sensitive features; the proposal is a stronger inductive bias, not an assertion that spectral information is completely absent.

**Safe implementation:** filter individual trials, or process appropriately separated continuous partitions with a justified boundary policy. Never concatenate unrelated trials for filtering. Use numerically stable filters and inspect edge effects on development data. Short epochs and low-frequency filters need special care. Keep all added bands inside the supplied 0.1–45 Hz passband; information above that cutoff cannot be recovered by a new architecture.

**Acceptance:** compare the spectral model alone and its addition to the fixed baseline ensemble using development predictions. A weaker standalone model is worth keeping only if the combined classifier improves. If this branch is competitive, joint waveform/power feature fusion is a second experiment—not part of the first ablation.

### 4.2 Second choice: residual dilated temporal convolutions

**Hypothesis:** nonlinear relationships between distant portions of the imagined-writing sequence may matter more than additional local filters.

The first kernel size is not the model's entire temporal window. With the documented DCN's three kernel-15, pool-2 blocks, the theoretical local receptive field before flattening is **106 input samples, approximately 424 ms at 250 Hz**, with output stride eight. This is calculated support, not a measurement of the learned effective receptive field. The final classifier nevertheless reads features from the whole trial. EEGNet already has whole-trial dependence through CBAM summaries and nonlinear cross-bin processing in its MLP. Therefore, a 60 ms first kernel does **not** imply that the model cannot recognize a component occurring at 300 ms.

The opportunity is **shared, temporally structured nonlinear processing of distant features**, rather than merely increasing the first convolution's kernel or giving these models their first access to global context. This adds long-lag nonlinear processing before DCN's linear readout; for EEGNet, it offers a different inductive bias from global summary gates and a position-specific MLP.

```text
Existing temporal/spatial convolutional stem
  -> modest channel projection
  -> residual temporal blocks, kernel 3, dilations 1 / 2 / 4 / 8
  -> position-preserving or coarse ordered-bin readout
  -> 26 logits
```

Use 32–64 feature channels initially. In one concrete DCN-based version, four blocks with two temporal convolutions each expand the local receptive field from 106 to **586 input samples, approximately 2.34 seconds**, without additional temporal striding. This is an analytical example; an EEGNet-stem version needs its own receptive-field calculation.

EEG-TCNet provides a primary-source precedent for compact dilated temporal processing [3]. Adapt the principle rather than copying its motor-imagery hyperparameters or aggressive pooling unchanged. Symmetric padding is legitimate for complete offline trials; use causal blocks if the intended deployment is online.

**Why this remains open:** adding generic multi-head attention, changing local kernel sizes, or reducing DCN filter counts does not test a small residual temporal model with an explicit receptive-field target.

**Acceptance:** compare against a stem/head-matched non-dilated control. Preserve ordering in the readout initially; do not combine this experiment with removal of all temporal position information. Watch training-versus-validation error for additional overfitting.

### 4.3 Third choice: compress the classifier without erasing timing

**Hypothesis:** excessive head capacity may be a better place to regularize than shrinking the convolutional feature extractor.

For the documented 24-channel, 801-point architecture:

- DCN's `Linear(8000, 26)` contains **208,026 parameters**, approximately 75% of its reported 278,246 total.
- EEGNet's `1024 -> 128 -> 26` classifier contains **134,554 parameters**, approximately 86% of the reported k=15 model total.

These counts do not prove overfitting. They identify a targeted experiment that differs from reducing filters throughout the network.

For DCN, first try:

```text
80 features x 100 time positions
  -> shared bias-free 1x1 projection, 80 -> 32 features
  -> flatten all 100 positions
  -> Linear(3200, 26)
```

This replacement has **85,786 parameters**, versus 208,026 in the current classifier, while retaining every output time position. It imposes a shared feature bottleneck; that restriction may also remove useful information, so retain the unchanged head as the control.

For EEGNet, a direct `1024 -> 26` classifier removes the hidden MLP without changing the temporal bins. Alternatively, test a smaller hidden layer as a separate comparison, not another simultaneous change.

**Why the prior adaptive-pooling head experiment does not settle this:** stronger time pooling changes temporal resolution as well as parameter count. The proposed bottleneck primarily changes feature/head capacity while preserving alignment. Start there before global average pooling, which could suppress useful cue-locked timing.

### 4.4 Fourth choice: distinguish early and late portions of the trial

The displayed letter is itself class-specific. **[INFERENCE]** early EEG may contain useful visual-cue information while later activity may contain different imagery-related information. This dataset does not, by itself, establish that either source dominates.

Begin with development-only window ablations, not a more complex network:

- Full epoch reference.
- Early window, for example 0–0.6 seconds.
- Later window, for example 0.6–2.8 seconds.

These are hypotheses about useful windows, not hard biological boundaries. Construct windows using the recorded time vector, not guessed sample indices. If distinct windows prove useful, use a small shared-stem/two-readout model or separately trained window specialists with fixed late fusion. Do not add a learned confidence gate initially; it introduces another noisy selection problem.

Using the same trial's predetermined windows at inference remains single-trial classification. Crops are correlated views, not additional independent examples. Train for the chosen windows; do not assume that shift-based TTA on an already timing-sensitive model is equivalent.

For a genuinely window-specific ablation, do not normalize a crop using statistics from the entire epoch: that would introduce information from the excluded phase. Filtering can also move information across window boundaries. Unknown upstream filtering therefore limits physiological localization even when the added processing is careful.

**Scientific limit:** cue-driven EEG information is not test-label leakage, but it does not demonstrate cue-independent handwriting imagery decoding. A late-window result would not prove purity either. If that distinction matters, future recordings need a design that separates cue identity from the imagery interval. A pre-cue-only development control can help flag acquisition confounds, although overlapping epochs and noncausal filtering complicate its interpretation.

### 4.5 Lower priority: a small convolutional transformer

A transformer is not ruled out by the historical self-attention attempt, but it is not the first experiment I would run on this dataset.

If the TCN motivates further investigation, try a convolutional stem producing a short sequence of temporal tokens, followed by only one or two pre-normalized transformer blocks with modest embeddings and explicit temporal-position information. Compare with a similarly sized TCN, not just the unchanged baseline. EEG Conformer supplies a relevant convolution-plus-attention precedent [4].

The risk is variance: 7,800 trials from one participant are not equivalent to 7,800 independent subjects or recording conditions. Bigger attention models, learned electrode graphs, and foundation-model fine-tuning add substantial assumptions and tuning opportunities. None is presently supported strongly enough here to outrank the simpler proposals.

## 5. Non-architectural improvements with a clear rationale

### 5.1 Prespecified multi-seed averaging, not seed selection

Training the same recipe under a small fixed list of seeds and averaging **all** its predictions is legitimate variance reduction. Selecting whichever seed has the highest test score is not.

For example, prespecify seeds 41, 42, and 43 for development comparisons. Keep split membership fixed; give each model an isolated random stream. Report mean performance and paired changes across the same folds/seeds. Multiple initializations are not cross-validation and do not supply independent subject-level evidence.

A multi-seed DCN or EEGNet ensemble is worth comparing against adding another similar architecture. Extra training and inference cost is the tradeoff. Do not require every seed to improve: use a predefined aggregate decision rule and inspect instability rather than cherry-picking winners or demanding an unrealistic universal win.

### 5.2 Evaluate ensemble contribution, not just standalone accuracy

Save development predictions by stable trial ID. Measure:

- Accuracy and negative log-likelihood of the actual combined predictor.
- Which errors the proposed member corrects, and which previously correct predictions it breaks.
- Prediction disagreement and paired error overlap.

Disagreement alone is not valuable; a poor classifier can disagree frequently without helping.

Start with the existing fixed mixture as a reference. Compare probability averaging with logit averaging as a small, declared experiment. Logit scales differ between architectures, label-smoothing policies, and SWA choices, so nominal 5:5:1 weights do not necessarily imply those relative influences.

If learning weights, use only a few nonnegative global weights, not per-class weights or a large stacking classifier. Generate honest out-of-fold predictions: each trial must be predicted by a model whose fitting and early-stopping process excluded that outer fold. Weight fitting still uses labels, so evaluate the fitted mixture on a separate development confirmation block or through nested folds—not on the same predictions used to fit it. Temperature fitting follows the same rule. Given prior attempts, this is a controlled secondary experiment, not the main recommendation.

### 5.3 Revisit normalization or resampling only for a specific reason

- Do not automatically z-score each band/window to unit variance in a power-based model: that can erase the proposed signal. Check provider normalization scope before adding another normalization layer. Fit any dataset-wide scaling within the training fold.
- Replacing BatchNorm with GroupNorm/LayerNorm is a possible training-only response to demonstrated normalization instability, not a guaranteed distribution-shift fix. It can remove useful amplitude information; defer it until the SWA/BN comparison is interpretable.
- Current downsampling is simple slicing, `raw_data[:, :, ::downsample_factor]` (`src/train.py:88–92`), not explicit anti-aliased resampling. At factor five, the new Nyquist limit is 25 Hz despite the documented input passband extending to 45 Hz. Historical downsampling attempts therefore do not establish a pure sampling-rate effect. If revisiting resolution, use anti-aliased resampling and scale kernels/windows in milliseconds. Retain 250 Hz as the initial reference.
- If revisiting covariance/Riemannian features later, use shrinkage or another explicitly positive-definite covariance estimate: provider-documented component removal means full spatial rank cannot be assumed. This is a concrete reason to scrutinize the earlier implementation, not evidence that a retry will beat the CNNs.

### 5.4 More independent data may matter more than another model family

If collection is possible, prioritize additional recording sessions with counterbalanced class order, session/event metadata, and a reserved future-session test set. More copies or synthetic transformations of existing epochs cannot substitute for new recording conditions.

Pretraining on genuinely separate EEG data may eventually help, but requires compatible channels, preprocessing, task transfer, and verified absence of evaluation-dataset overlap. Repeating the failed contrastive recipe or adding a large pretrained encoder is not a first-line recommendation without that evidence.

## 6. A bounded experiment plan

Do not test every architecture/optimizer/augmentation combination. Use the same development folds and seeds, change one mechanism at a time, and keep the original model as a reference.

| Stage | Comparison | Decision |
|---|---|---|
| A | Explicit baseline recipes; isolated RNG; frozen inference; correctly identified saved states | Establish a reproducible development reference without querying test accuracy |
| B | Best checkpoint vs parameter averaging plus clean training-only BN refresh | Select the actual prediction model on development data; no assumed SWA advantage |
| C | Compact spectral model alone and added to the reference ensemble | Keep only a reproducible development gain, standalone or in combination |
| D | Residual dilated TCN vs matched non-dilated comparator | Test long-range nonlinear temporal interaction |
| E | Position-preserving smaller head vs unchanged head | Test head capacity without simultaneously changing temporal pooling |
| F | Early/late-window ablations; specialize only if justified | Determine whether a two-window architecture earns its complexity |
| G | Prespecified seed averaging and a small ensemble-combination comparison | Retain additional inference cost only for useful development performance |
| Final | Freeze the recipe, then evaluate on the fixed test or a genuinely untouched future-session test | Report the result once; do not feed it back into another tuning cycle |

For each stage, retain configuration, split/group IDs, software/backend, model seeds, actual checkpoint type, development predictions, and uncertainty in paired differences. Use group/block-level resampling for uncertainty when trials are dependent. An independent-trial paired test such as McNemar's is appropriate only when its independence assumptions are credible. Do not treat seed repetitions of the same trials as extra independent test observations.

After selection, optionally refit on all development data using a training-duration rule fixed from the development folds. Recompute any final BN statistics on that fitting data, never on test data. Decide this refitting policy before inspecting final performance.

**Stop rule:** retain a change for a consistent, useful aggregate development improvement, not a single favorable run. When differences remain within substantial uncertainty, prefer the simpler model or collect new sessions. There is no defensible basis here for promising a particular final percentage.

## 7. What I would not prioritize

- Another broad sweep of AdamW learning rates, dropout, label smoothing, optimizer brands, or nearby temporal kernels before fixing comparison confounds.
- A fourth similar EEGNet variant chosen because its test result happens to help.
- Repeating SE/CBAM additions, generic attention, or confidence gating without a new, explicit hypothesis.
- Test-set BN adaptation, test-time parameter updates, label-quota matching, or exploitation of trial order.
- Heavy stacking on the same 780 validation predictions used to judge the stack.
- Whole-trial averaging across repeated instances of the same unknown test letter, which changes the single-trial problem or uses unavailable label grouping.
- Treating unrelated four-class motor-imagery paper accuracies as expected performance on 26 imagined letters.

**My first substantive new architecture would be the compact filter-bank/log-variance model. My first alteration to the current CNN family would be a small residual TCN or a position-preserving smaller head.** Which wins remains an empirical question; the existing history does not close these avenues.

## 8. Sources and evidence boundaries

Repository evidence: [`process.md`](process.md), [`README.md`](README.md), [`src/train.py`](src/train.py), [`src/extract.py`](src/extract.py), [`models/deep_conv_net.py`](models/deep_conv_net.py), [`models/eegnet.py`](models/eegnet.py), [`models/eeg_inception.py`](models/eeg_inception.py), and the provider's [`data_instruction.docx`](data/raw/data_instruction.docx), especially its Data section. MAT/NPZ inspection was limited to schema and channel/time metadata; no EEG or class-label arrays were analyzed. Historical performance claims are excluded from the accepted baseline. Existing-model parameter counts were checked by constructing the specified models without EEG data. Receptive-field and proposed-head counts are analytical calculations for the specified 24-channel, 801-point configurations; constructor defaults and other sampling rates can differ.

1. Mane et al., **FBCNet: A Multi-view Convolutional Neural Network for Brain-Computer Interface**. [Paper](https://arxiv.org/abs/2104.01233); [authors' implementation](https://github.com/ravikiran-mane/FBCNet), including spatial filtering and the log-variance layer. Supports the spectral-power architecture hypothesis, not a handwriting-accuracy forecast.
2. Schirrmeister et al., **Deep learning with convolutional neural networks for EEG decoding and visualization**. [Full paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5655781/). Describes shallow power-estimation networks, deep networks, and cropped training. Results on its tasks do not guarantee transfer here.
3. Ingolfsson et al., **EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery Brain-Machine Interfaces**. [Paper](https://arxiv.org/abs/2006.00622); [authors' implementation](https://github.com/iis-eth-zurich/eeg-tcnet). Supports compact residual dilated temporal processing.
4. Song et al., **EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization**. [Authors' repository and paper link](https://github.com/eeyhsong/EEG-Conformer). Supports the lower-priority convolution-plus-attention option.
5. PyTorch, **AveragedModel and BatchNorm statistics for SWA/EMA**. [Documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.swa_utils.AveragedModel.html); [implementation of averaging and `update_bn`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/optim/swa_utils.py). Supports explicit averaging/buffer handling and training-only recalibration as a controlled experiment.
