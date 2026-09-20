# DiffKT — reproducibility package

Code accompanying the manuscript *Robust Knowledge Tracing via Bidirectional
Context and Conditional Diffusion* (Scientific Reports).

The package contains the **data preprocessing scripts**, the **baseline model
implementations**, the **training and evaluation scripts**, and the **exact
hyperparameters of every baseline** in the comparison tables. The DiffKT method
implementation (`bikt_pres.py`, `diff_prelstm_as2.py`, `torch_kt/diff_utils.py`)
is not part of the deposit; it is subject to institutional intellectual-property
restrictions, as stated in the Code availability section of the manuscript.

---

## 1. Code structure

```
code/
├── run.py                     training script for all baselines
├── run_for_evaluate.py        evaluation script (AUC / ACC, R-AUC / R-ACC)
├── run_for_predict.py         export per-step predictions from a trained run
├── requirements.txt           environment
├── configs/                   DiffKT configuration templates (Stage 1 / Stage 2)
└── torch_kt/
    ├── preprocesss/           dataset construction, one script per dataset
    ├── models/                baseline model implementations
    ├── data_loader.py         learner-level split and k-fold expansion
    ├── training.py            base trainer shared by all models
    ├── runner.py              model registry and training loop
    ├── metrics.py             AUC, ACC and response-reversal metrics
    ├── efficiency_monitor.py  parameter count / latency / throughput
    └── backbone_models.py     shared attention and embedding blocks
```

**Preprocessing scripts** — `torch_kt/preprocesss/<dataset>_preprocess.py` build
the four datasets of the paper from the public raw files. Each module exposes
`process_raw_data(raw_file, out_dir)`, which writes
`data/<dataset>/{all.pkl, info.json, skill_response.json, problem_response.json}`;
`<dataset>` is the value passed as `--data_name` when training.

| Dataset in paper | Script | Dataset directory |
|---|---|---|
| ASSIST2009 | `assist0910_preprocess.py` | `assist0910` |
| ALG2005 | `algebra2005_preprocess.py` | `algebra2005` |
| B2A2006 | `bridge2algebra2006_preprocess.py` | `b2algebra2006` |
| NIPS/Eedi | `nips_preprocess.py` | `nips` + sub-sample suffix (default `34`) |

The learner-level train/validation/test partition and the cross-validation folds
are then created by `torch_kt/data_loader.py` on first use.

**Baseline models** — one file per model in `torch_kt/models/`. The class name is
the model name passed to the training script (e.g. `DKT`, `AKT`, `DTransformer`).

**Training script** — `run.py`, entry point for every baseline.

**Evaluation scripts** — `run_for_evaluate.py` recomputes AUC/ACC and the
response-reversal robustness metrics (R-AUC, R-ACC); `run_for_predict.py` exports
per-step predictions. Both read the log directory written by a training run.

---

## 2. Usage

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
cd code                                             # so that torch_kt is importable
```

```bash
# 1. build a dataset from the public raw files -> data/<dataset>/
#    (call process_raw_data in the corresponding script under torch_kt/preprocesss/)

# 2. train a baseline (5 folds, sequence length 100)
python run.py DKT assist0910 --data_base ../data --logs_base ../logs --folds 5 --max_len 100

# 3. evaluate
python run_for_evaluate.py ./logs --data_base ../data
python run_for_predict.py ./logs/<run_dir> --data_base ../data --result_base ./predict_result
```

Reference environment: Python 3.8, PyTorch 2.0.1, CUDA 11.x, one NVIDIA RTX
A6000 (48 GB). Every run writes its own `run_config.json`, its curves
(`train.csv`), its test metrics (`test_result.csv`) and its best checkpoint
(`weights/model.pth`) into `logs/<run_dir>/`.

---

## 3. Baseline hyperparameters

**Shared training protocol (identical for DiffKT and every baseline).**
Learner-level 80 % train pool with a 20 % held-out tuning pool; 5 learner-level
folds inside the 80 % pool; sequence length 100; context embedding 64; hidden
state 128; Adam with learning rate 1e-3; weight decay 0; dropout 0.1; batch size
32; up to 100 epochs with early stopping (patience 5) on validation AUC; random
seed 100; knowledge component–response pairs `(c_t, r_t)` only, no question IDs
or difficulty annotations.

**Per-model settings** (the 15 baselines appearing in the manuscript tables).
Values below are the effective configuration; `hidden` = hidden state dimension.

| Model | Class | Effective hyperparameters |
|---|---|---|
| DKT | `DKT` | emb 64, hidden 128, dropout 0.1 |
| KQN | `KQN` | mlp_hidden 128, rnn_hidden 128, hidden 128, dropout 0.1 |
| DKVMN | `DKVM` | memory_dim 128, memory_size 64, dropout 0.1 |
| ATKT | `ATKT` | embed_size 128, hidden 128, answer_dim 8, attention_dim 64 |
| Deep-IRT | `DeepIRT` | memory_dim 128, memory_size 64 |
| ReKT | `ReKT` | emb 64 |
| SAINT | `SAINT` | emb 64, att_heads 4, attention_blocks 2 |
| SAKT | `SAKT` | emb 64, att_heads 4, attention_blocks 2 |
| AKT | `AKT` | emb 64, att_heads 4, attention_blocks 2, d_ff 256, final_fc_dim 256, l2 1e-5 |
| SparseKT | `SparseKT` | emb 64, att_heads 4, attention_blocks 2, d_ff 256, final_fc_dim 256, separate_qa false, l2 1e-5, k_index 16 |
| StableKT | `StableKT` | emb 64, att_heads 4, attention_blocks 2, d_ff 256, final_fc_dim 256, separate_qa false, l2 1e-5, penumbra_r 0.1, penumbra_g 0.1 |
| DTransformer | `DTransformer` | emb 64, att_heads 4, final_fc_dim 256, n_know 16, score_type q, lambda_cl 1.0, proj false, hard_neg true |
| SimpleKT | `SimpleKT` | emb 64, att_heads 4, attention_blocks 2, d_ff 256, final_fc_dim 256, kq_same true, separate_qa false, l2 1e-5 |
| LeftoKT | `LeftoKT` | emb 64, attn_heads 8, attention_blocks 2, d_ff 256, final_fc_dim 256, kq_same true, separate_qa false, l2 1e-5 |
| FlucKT | `FlucKT` | emb 64, attn_heads 8, attention_blocks 2, d_ff 256, final_fc_dim 256, kq_same true, separate_qa false, l2 1e-5, kernel_size 5 |

Two models keep an architecture-specific value that the shared protocol does not
override, because their class names the parameter differently from the
corresponding command-line flag: ATKT uses `embed_size=128` (the flag is
`--emb_size`) and LeftoKT / FlucKT use `attn_heads=8` (the flag is `--att_heads`).

A model's configuration is assembled by `torch_kt/runner.py` with last-wins
precedence: constructor defaults, then dataset metadata, then `best_config.json`,
then the `run.py` command line, then `max_len`. The flags `--dff`, `--n_layers`,
`--shortcut`, `--use_cl`, `--sparse_ratio`, `--num_buckets` and `--max_distance`
are declared in `run.py` but reach no model constructor, so their default values
have no effect on the reported results and should not be cited as model
settings.

---

## 4. Licence and citation

【Choose and add a licence before depositing — e.g. MIT or Apache-2.0 for code,
CC-BY-4.0 for the configuration files.】

If you use this code, please cite:

```bibtex
@article{diffkt,
  title  = {Robust Knowledge Tracing via Bidirectional Context and Conditional Diffusion},
  author = {He, Bo and Huang, Zhijun and Liu, Shengyingjie},
  journal = {Scientific Reports},
  year   = {【year】}
}
```
