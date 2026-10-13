# Hugging Face / Lightning / Hydra research template

A template for experiments that contain multiple model fits: cross-validation,
Optuna selection, refitting, binary or multilabel threshold fitting, single-model
and ensemble evaluation, and interpretation.

The research entry point includes **offline feature and Hugging Face text examples**.
Both run through the same CV, selection, threshold, ensemble, and interpretation
workflows. The Hugging Face backend keeps Arrow datasets intact, dynamically pads
tokenized batches, and trains Transformers sequence classifiers with Lightning.
The original Hugging Face entry points remain available separately.

## Installation

Use Python 3.10 or newer. Install PyTorch for your hardware first, then dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
# CPU example; use your appropriate PyTorch build for GPU research.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Run commands from the repository root. The examples need no dataset/model downloads
after installing dependencies. `environment.yaml` provides a Conda alternative.
Every workflow records the installed package versions. For a paper or release,
freeze your actual hardware-specific environment with `pip freeze > requirements.lock`
and commit that lock file with the experiment configs.

`requirements-test-cpu.lock` records the Linux/Python 3.12 CPU environment used for
the integration tests. Install it with `pip install -r requirements-test-cpu.lock`
to reproduce that environment; notebook extras are not included. It is an environment
snapshot, not a universal GPU lock.

## First experiment

```bash
# Three Optuna trials, each with five-fold CV, followed by the selected model refit.
# Test evaluation is disabled by default.
python -m src.research_cli output_dir=logs/my_experiment

# Explicit final test evaluation of the frozen predictor.
python -m src.research_cli action=evaluate output_dir=logs/my_experiment

# Interpret development examples without retraining.
python -m src.research_cli action=interpret output_dir=logs/my_experiment
```

The default protocol reserves 20% of samples for testing and 20% of the remaining
samples for decision fitting. CV and refitting use the remaining development
partition. The calibration partition is never used to train model parameters.
Thresholds are fitted on the final predictor's calibration probabilities, including
the actual combined probabilities when using an ensemble.

```text
Data → development / calibration / test partitions
          │                 │             │
      Optuna study          │             │
          │                 │             │
      Trial → fold fits → mean validation objective
          │                 │             │
      Select configuration  │             │
          │                 │             │
      Refit predictor ──────┘             │
          │                               │
      Fit thresholds → freeze predictor ──┘
                              │
                       evaluate / interpret
```

CV selection scores are not independent final performance estimates. Test results
must not guide selection. If test interpretation informs new model changes, treat
that test set as development data thereafter.

## Hugging Face research backend

Run the same protocol using a tiny, randomly initialized BERT and an in-memory
Hugging Face dataset. These commands do not access the Hub:

```bash
python -m src.research_cli --config-name research_hf output_dir=logs/hf_example
python -m src.research_cli --config-name research_hf action=evaluate output_dir=logs/hf_example
python -m src.research_cli --config-name research_hf action=interpret output_dir=logs/hf_example \
  interpretation=integrated_gradients
python -m src.research_cli --config-name research_hf action=interpret output_dir=logs/hf_example \
  interpretation=attention
```

`research_hf` selects `research_backend=huggingface`, `research_data=hf_toy`,
`research_model=hf_tiny`, and `search=huggingface`. The feature example remains the
default `research` configuration. CV, nested CV, OOF threshold transfer, refitting,
single/seed/fold predictors, evaluation, and interpretation share the same orchestration.

For a pretrained Transformer and your own dataset:

```bash
python -m src.research_cli --config-name research_hf_pretrained \
  output_dir=logs/my_text_project \
  data.path=your-org/your-dataset data.text_column=text data.label_column=label \
  data.id_column=sample_id model.name_or_path=distilbert/distilbert-base-uncased
```

Repeat those data/model overrides for later actions, or save them in a project YAML
config inheriting `research_hf_pretrained` and use its config name for every command.
`data.path` also accepts a local dataset loader such as `json` with `data.data_files`,
or a saved Arrow dataset directory with `data.from_disk=true`. Set `data.cache_dir`
to override the standard Hugging Face download/processing cache. The loader selects
one map-style `Dataset` through `data.split`; streaming datasets are unsupported.
If you need published train/test assignments, assemble the sample population in a
data factory and supply the published sample-ID split manifest. Do not silently
resplit a dataset that already has a required evaluation protocol.

Required columns are stable unique **string IDs**, text, and numeric targets.
Binary targets may be scalar 0/1 or length-one vectors. Multilabel targets are
0/1 vectors in the order specified by `data.labels`. Set `task=multilabel`,
`data.labels='[label_a,label_b]'`, and an appropriate split strategy for those tasks.
Optional `data.text_pair_column` enables sentence pairs; `data.group_column`
supplies groups for split isolation. For the offline multilabel example:

```bash
python -m src.research_cli --config-name research_hf output_dir=logs/hf_multilabel \
  task=multilabel data.n_labels=2 split.strategy=kfold decision=per_label predictor=seeds
```

The data adapter retains Arrow storage and uses `.select()` for partitions and
`.map()` for tokenization. Only ID/target/group metadata is converted to Python or
NumPy structures. Tokenization uses a fixed tokenizer vocabulary; it does not learn
from validation or test text. Dynamic padding happens per batch, with truncation
controlled by `model.max_length`. Lightning trains the model with AdamW and BCE
logit loss, including a **single sigmoid output for binary classification**.
This is separate from the original ESM example's two-class softmax convention.

`model.name_or_path` selects `AutoModelForSequenceClassification`.
`model.tokenizer_name_or_path` can select another compatible tokenizer.
Pin `model.revision`, `model.tokenizer_revision`, and `data.revision` to immutable
Hub commits for reproducibility. A local model directory can be used with
`model.local_files_only=true`. `model.config` accepts architecture configuration
overrides. An incompatible pretrained classification head fails by default; use
`model.ignore_mismatched_sizes=true` explicitly when replacing that head. Other
incompatible architecture shapes should be fixed in the model configuration.

Each fit exports a `huggingface/` directory containing model safetensors, model
configuration, tokenizer assets, and `metadata.json`. Metadata records label order,
input columns, truncation length, training/validation IDs, dataset fingerprint,
requested revisions, and resolved commit hashes when provided by Transformers.
Final manifests reference these directories. Reloading evaluation and interpretation
models uses **local saved artifacts only**, even when training started from the Hub.

HF gradient saliency takes the L2 norm of gradients over word-embedding dimensions
for each token. Integrated gradients sums embedding attributions per token, using
zero word embeddings as the baseline while retaining masks, positions, and segment
IDs. Both explain the chosen pre-threshold label logit and zero padding positions.
Attention analysis averages layers/heads and masks padded query/key positions.
Attention-capable architectures are loaded with eager attention for inspection.
Token IDs, token strings, masks, and sample IDs are saved with each interpretation;
ensemble attribution averaging rejects different token mappings. Tokens are not
automatically mapped back to protein residues or original character spans.

This backend covers binary and multilabel **text/sequence classification** using
Transformers models that expose sequence logits. Multiclass, token classification,
generation, vision/audio processing, quantized/PEFT training, and distributed fits
need additional task-specific adapters. A real pretrained Hub model/GPU is not part
of the offline verification; tests exercise actual tiny Transformers models,
including loading a local pretrained checkpoint, without network requests.

## Commands and configuration

`configs/research.yaml` composes independently configurable data, model, split,
backend, optimization, search, predictor, decision, and interpretation groups.

| Action | Behavior |
| --- | --- |
| `cv` | Evaluate the supplied configuration through folds; save OOF predictions |
| `search` | Run/resume Optuna selection without refitting or testing |
| `refit` | Refit the saved selected configuration and fit its decision policy |
| `run` | Search, select, refit, and fit the decision policy |
| `evaluate` | Evaluate the saved frozen predictor on the saved test partition |
| `interpret` | Run a method plugin on selected examples of a frozen predictor |
| `nested_cv` | Run independent inner studies within outer evaluation folds |

```bash
python -m src.research_cli action=cv output_dir=logs/baseline
python -m src.research_cli action=search output_dir=logs/search_a search.n_trials=20
python -m src.research_cli action=refit output_dir=logs/search_a
python -m src.research_cli action=evaluate output_dir=logs/search_a

# A separate outer study per fold; substantially more expensive.
python -m src.research_cli action=nested_cv output_dir=logs/nested
```

Use the same data/split settings for subsequent commands. `final_test=true` explicitly
enables testing at the end of `run` or `refit`. Nested CV uses the entire supplied
dataset as its outer evaluation population; it does not additionally reserve a
separate test set. Do not supply an external test set as part of that population.

Refit duration is the median selected CV epoch count, rounded down and bounded at
one epoch. There is no early stopping on final test data. The example selects fold
checkpoints using validation BCE loss and selects trials using mean fold average
precision. These are deliberately separate selection criteria.

### Optuna and recovery

The native Python Optuna runner owns the fold loop. The old Hydra Optuna plugin
and MNIST search example were removed because that plugin pins Optuna below 3.
Hydra still controls all experiment settings.

`search.parameters` maps dotted config paths to `float`, `int`, or `categorical`
specifications. Float parameters support `log: true`. The default objective is
mean fold macro average precision; all supported search objectives are maximized.
Binary AUROC is another option. Threshold-dependent scores currently use a fixed
0.5 cutoff during selection; final threshold fitting happens afterward.

The SQLite study is stored under `output_dir/study.db`, or configure
`search.storage` for an external database. `search.n_trials` is a total budget,
including failed/pruned trials, and can be increased on a later command.
Pruning compares running mean scores after completed folds using distinct step
numbers. Completed matching fits are reused; changed data, split manifests, or
training settings are rejected.
The implementation source fingerprint is also checked before reusing fits/studies.

Use one workflow process per output directory. Automatic recovery of a process
killed mid-trial is not provided: Optuna can retain a RUNNING trial. Inspect and
mark that trial failed through Optuna before increasing the budget and continuing.
Completed fit recovery applies to rerunning the same CV/fit paths; a new Optuna
trial receives a new directory. Mid-epoch optimizer-state resume is not implemented
in the feature backend.

### Binary and multilabel thresholds

```bash
# A per-label multilabel example; use non-stratified KFold for this example.
python -m src.research_cli output_dir=logs/multilabel \
  task=multilabel data.n_labels=3 split.strategy=kfold decision=per_label
```

Decision modes are `fixed`, `global`, and `per_label`. Global thresholds can optimize
macro/micro F1 or binary MCC. Independent per-label fitting supports macro F1;
rare labels fall back to the configured threshold. The grid and minimum positive/
negative support are configurable. Saved fitting scores are explicitly tuning
scores, not independent performance estimates. Probability calibration is distinct
from threshold fitting and is not included in the initial methods.

To refit a **single model on all non-test samples**, transfer thresholds fitted on
the selected trial's out-of-fold predictions:

```bash
python -m src.research_cli output_dir=logs/full_refit \
  decision=oof split.calibration_fraction=0
python -m src.research_cli action=evaluate output_dir=logs/full_refit \
  decision=oof split.calibration_fraction=0
```

OOF thresholds may transfer imperfectly because the refit's score distribution can
change. This mode supports single refits only. Ensembles require held-out calibration:
one OOF score per sample does not reproduce a multi-model average at inference time.

### Single models and ensembles

```bash
python -m src.research_cli output_dir=logs/seed_ensemble predictor=seeds
python -m src.research_cli output_dir=logs/fold_ensemble predictor=folds
```

`single` refits one seed. `seeds` refits each seed and averages probabilities.
`folds` reuses the selected trial's fold checkpoints without full-data refitting.
All use the same prediction/evaluation interface. `predictor.weights` accepts
explicit nonnegative weights, otherwise averaging is uniform. Weight learning,
stacking, and automated heterogeneous-model selection are extension points.

Frozen predictors reject replacement by different decision/predictor settings.
To compare another predictor from the same saved study, use `action=refit` and a
different `final_name`, for example `final_name=seed_ensemble predictor=seeds`.
Use that `final_name` when evaluating or interpreting it.

## Interpretation plugins

```bash
python -m src.research_cli action=interpret output_dir=logs/my_experiment \
  interpretation=integrated_gradients interpretation.target=0
```

For numeric features, included methods are signed gradient saliency and integrated
gradients with a zero baseline in raw feature coordinates. For HF sequences they
dispatch to the embedding methods described above. Each analysis writes its own UUID directory.
The target is a pre-threshold label logit. Ensemble outputs are weighted means of
member-logit attributions, **not** attributions of the combined probability.

`interpretation=attention` supplies a capability-checked attention plugin. Its model
adapter must expose `get_attentions(inputs)` returning a tuple of tensors shaped
`[samples, heads, tokens, tokens]`. It exports the mean across layers and heads.
The example MLP has no attention and will reject that method. Attention patterns
are descriptive outputs rather than evidence of causal importance.

Add project-specific methods using this interface:

```python
class MyMethod:
    def explain(self, model, inputs, target):
        # Return an array with a leading sample dimension.
        ...
```

Select the class through a Hydra `_target_`. The runner enables method-specific
gradient behavior and retains checkpoint, method, target, sample, and environment
provenance. Input-to-residue/token mappings, baselines, plots, and model-specific
capabilities belong in project adapters and methods.

## Adapting to your project

Choose the feature backend for numeric inputs and the Hugging Face backend for
text/sequence classifiers. The original custom ESM wrapper remains a separate
entry point; the new backend uses standard Transformers sequence-classification models.

For precomputed features, use `research_data=npz data.path=/absolute/path/data.npz`.
The NPZ must contain numeric arrays `x[N,D]` and `y[N,L]`, Unicode arrays `ids[N]`
and `labels[L]`, and optional `groups[N]`. Object/pickle arrays are rejected.

1. Add a data factory selected by `research_data`. Return `ResearchData` for numeric
   features or `HuggingFaceData` for Arrow-backed text. Both expose binary targets
   `y[N,L]`, stable string sample IDs, label names, optional groups, a fingerprint,
   validation, and `select_inputs(indices)`. The feature backend expects numeric `x[N,D]`.
2. Add a model config under `research_model`; the feature backend instantiates it
   with `input_dim` and `output_dim` and expects logits `[N,L]`.
3. For other modalities, select a `research_backend` config implementing:

   ```python
   fit(cfg, data, train_ids, validation_ids, seed, output_dir)
   # -> checkpoint path + selected_epochs; train and validation must be disjoint
   load(checkpoint)  # -> model/adapter
   predict(model, data, indices)  # -> probabilities [N,L], in indices order
   # Optional: explanation_inputs(model, data, indices) -> structured method inputs
   ```

4. Fit preprocessing using training IDs only. Preserve its state with checkpoints.
   The reference backend saves training-only feature normalization in model buffers.
5. Add interpretation adapters and methods for your model family.

Both supplied backends run one CPU or GPU device per fit. Distributed training,
residue/character attribution mapping, multilabel stratification, and temporal
splitters require corresponding project adapters. Built-in splitting supports binary
stratification, ordinary KFold, and binary stratified group folds. Group holdouts
use GroupShuffleSplit to prevent group overlap.

For published splits, temporal splits, or project-specific multilabel stratification,
set `split.manifest_path=/absolute/path/splits.json`. This JSON contains
`development`, `calibration`, and `test` lists of sample IDs plus `folds`, a list
of objects with `train` and `validation` ID lists. The loader validates disjoint
partitions, complete coverage, and one validation occurrence per development sample.
With `split.strategy=group`, it also checks group isolation. The supplied file's
content participates in the split fingerprint.

Record each hypothesis, reference condition, changed factors, and reporting status
in versioned experiment configs or a research log. Keep raw data immutable and
version preprocessing/split definitions. Git branches track code changes; configs
track experimental conditions. Final claims should include multiple independently
trained seeds rather than only a best seed.

## Artifacts and analysis

```text
logs/my_experiment/
├── config.yaml, environment.json, splits.json
├── study.db, selection.json
├── trial-0/
│   ├── config.yaml, cv.json, oof.parquet, oof_folds.json
│   └── fold-0/  # model.pt, checkpoints/, result.json, status.json, validation.parquet
├── final/
│   ├── seed-0/
│   ├── decision_fit.parquet
│   └── predictor.json
├── evaluation/
│   ├── test.parquet
│   └── metrics.json
└── interpretation/<analysis-id>/
    ├── attributions.parquet
    └── manifest.json
```

Prediction Parquet files contain `sample_id`, `fold`, `target`, and probability
`score` vectors. A JSON sidecar records label order and score type. Pooled OOF files
use the separate `oof_folds.json` mapping for fold membership. Never combine models
using implicit row order or different label schemas.

Manifests record content fingerprints, sample assignments, resolved configs,
checkpoint references, thresholds, seeds, Git state, and installed package versions.
Keep checkpoint paths intact when moving artifacts, or update the manifest paths.
Load checkpoints only from trusted sources. Separate output directories identify
different protocols; frozen model files are not overwritten with changed settings.

Aggregate explicit compatible final evaluations:

```bash
python -m src.analyze logs/seed0/evaluation/metrics.json \
  logs/seed1/evaluation/metrics.json --output logs/summary.json
```

Aggregation retains individual values and reports means and sample standard
deviations. It rejects incompatible data/splits/label schemas and duplicate paths.
For replications on the same test partition, keep `seed` (the split/CV seed) fixed
and vary `predictor.seeds` across refits using different `final_name` values.
Use distinct `evaluation_name` values when evaluating different predictors from
one study (the default is `evaluation`). Existing evaluations reject replacement
by another predictor. Confidence intervals,
paired statistical tests, and publication plots require a design-specific analysis.

## Original Hugging Face entry points

`python src/train.py experiment=example` and `python src/eval.py ckpt_path=...`
remain for the original sequence-classification path. Supply a Hugging Face dataset
with `train`, `validation`, and `test` splits and `sequence`/`label` columns, together
with a matching label mapping. The example config names, model output pooling,
processor injection, metrics signature, data-module lifecycle, and validation loss
were corrected. Training defaults to `test=false` and accepts seed zero. Hub export
updates repositories without deleting their existing history.

The offline tests do not download or validate a real ESM checkpoint. Token-level
classification needs its own head, target layout, loss, metrics, and backend.

## Verification

```bash
pytest
```

The obsolete MNIST tests were replaced with offline integration checks covering
search/CV/refit/evaluation, single and ensemble predictors, binary/multilabel decisions,
split/group isolation, preprocessing isolation, artifact recovery, nested CV, and
interpretation against a known linear model. The reference example is intended to
make protocol changes testable before launching expensive project runs. Additional
HF checks exercise real tiny BERT fits, dynamic padding, local pretrained loading,
binary/multilabel ensembles, saved tokenizer reuse, and token-level interpretation.
