# aircheck-benchmark-pipeline

Runs every model of the AIRCHECK benchmark on the same data and writes one report comparing them.

It trains nothing itself. Each model lives in its own repository with its own environment, and each already
knows how to run itself ([MMELON](https://github.com/ShagReza/biomed-multi-view),
[MAMMAL](https://github.com/ShagReza/biomed-multi-alignment): `aircheck/benchmark/run_all.sh`). This pipeline
does the four things one repository cannot:

1. Makes the cross-validation folds **once**, so every model is tested on exactly the same molecules.
2. Runs each model **in its own environment**, because they need different versions of PyTorch and cannot share one.
3. Puts every model's runs in **one folder**, without touching the repositories' own `output/`.
4. Writes **one report** over all of them, with [aircheck-benchmark-core](https://github.com/ShagReza/aircheck-benchmark-core).

```
<results_root>/<experiment>/
├── folds/                  made once, read by every model
├── mmelon/                 one folder per model
│   ├── finetune_multiview/     a run: predictions.csv, run.json, progress.log, models/
│   ├── lgbm_pretrained_fused/  ...
│   └── embeddings/             shared by this model's LightGBM runs
├── mammal/
│   ├── finetune_lora/
│   └── ...
├── report.md               every model's runs, scored the same way
├── report_by_model.csv     the best run of each model on the test set
└── .pipeline/              one log and marker per model
```

## Start here: from nothing to a report

Four repositories, then one command. Everything below assumes `~/repos` and `~/envs`; use your own paths if
you prefer.

**1. Clone them**

```bash
mkdir -p ~/repos && cd ~/repos
git clone git@github.com:ShagReza/aircheck-benchmark-pipeline.git   # this one: runs the benchmark
git clone git@github.com:ShagReza/aircheck-benchmark-core.git       # folds, metrics, report (shared)
git clone git@github.com:ShagReza/biomed-multi-view.git             # MMELON
git clone git@github.com:ShagReza/biomed-multi-alignment.git        # MAMMAL
```

**2. Build one environment per model** — they need different versions of PyTorch and cannot share one. Each
fork's `aircheck/INSTALL.md` has the steps, including installing `aircheck-benchmark-core` into it:

```bash
# MMELON  -> ~/envs/biomed-multiview     see ~/repos/biomed-multi-view/aircheck/INSTALL.md
# MAMMAL  -> ~/envs/biomed-mammal        see ~/repos/biomed-multi-alignment/aircheck/INSTALL.md
```

**3. Build this pipeline's own environment** (Snakemake plus core; it never loads a model):

```bash
cd ~/repos/aircheck-benchmark-pipeline
micromamba create -y -f envs/pipeline.yaml -p ~/envs/aircheck-pipeline
micromamba activate ~/envs/aircheck-pipeline
```

**4. Say where your data and models are**, in `config/`:

- `config/datasets.yaml` — your train and test files, and their column names.
- `config/config.yaml` — the experiment to run (which dataset, which fold method) and where results go.
- `config/models.yaml` — each fork's folder and environment, if you did not use the paths above.

**5. Run it**

```bash
snakemake check --cores 1    # do the environments work?
snakemake -n                 # the plan, running nothing
snakemake --cores 4 --resources gpu=1
```

You get `<results_root>/<experiment>/report.md`: every model and method, scored identically, with
`report_by_model.csv` naming the best run of each model.

**What happens inside:** the pipeline makes the cross-validation folds once, then calls each fork's
`aircheck/benchmark/run_all.sh` in that fork's environment, giving it those same folds. Each run writes only
`predictions.csv` and `run.json`; the pipeline scores all of them together at the end. The models never
compute their own metrics, which is what makes the comparison fair.

**Adding your own model** means a fork with an `aircheck/benchmark/run_all.sh` that writes those two files,
and one entry in `config/models.yaml`. Nothing else.

## Environments

The pipeline **uses** environments; it never creates or changes them. A model's environment is built once from
its fork's `aircheck/INSTALL.md`, because those steps are model-specific (MMELON, for instance, needs a
particular compiler to build `pytorch-fast-transformers`). `config/models.yaml` records where each one is, and
`snakemake check` confirms each can import its model and core before a long run starts.

Only the pipeline's own environment is automatic: it holds Snakemake and core and never loads a model, so
`snakemake --use-conda` can build it from `envs/pipeline.yaml` instead of the `micromamba create` above.

## Run

```bash
snakemake check --cores 1          # do the configured environments work?
snakemake -n                       # the plan, without running anything
snakemake --cores 4                # run it
snakemake --cores 4 --resources gpu=1   # one GPU model at a time
```

- **Resumable.** Each fork skips runs it has already finished, and Snakemake skips models that are done, so
  starting it again after an interruption carries on. On a Workbench VM that shuts down after a few hours,
  simply start it again.
- **Long runs:** `nohup snakemake --cores 4 --resources gpu=1 > pipeline.log 2>&1 &`, then follow
  `<results_root>/<experiment>/.pipeline/<model>.log` for a model's own progress.
- **One experiment:** `snakemake <results_root>/wdr91-cluster/report.md`.

## Settings

| File | What it holds |
|---|---|
| `config/config.yaml` | where results go, which experiments to run (dataset plus fold settings), which models |
| `config/models.yaml` | one entry per fork: its folder, its environment, which of its runs to do |
| `config/datasets.yaml` | one entry per dataset: the train and test files and their column names |

**Adding a model** is one entry in `config/models.yaml`: the fork's folder and environment. The fork must have
`aircheck/benchmark/run_all.sh`, which writes `predictions.csv` and `run.json` per run — that is the whole
contract, and core does the rest.

**Adding a dataset** is one entry in `config/datasets.yaml` plus an experiment in `config/config.yaml`.

**Another condition on the same data** (other folds, other settings) is a new experiment name, so results
never mix.

## How it works

| Rule | What it does |
|---|---|
| `folds` | `aircheck-bench make-folds` into `<experiment>/folds/` |
| `run_model` | `PATH=<env>/bin:$PATH <fork>/aircheck/benchmark/run_all.sh --experiment … --root … --folds … --no-report` |
| `report` | `scripts/collect.py`: core's report over every run, plus the best run per model |

`--folds` means every model reads the fold files this pipeline made, and `--no-report` leaves the report to the
final step. Core checks that each run's fold predictions are exactly those folds' test rows, so a run on other
folds cannot slip into the table.

## Notes

- **Disk.** A fine-tuning experiment is tens of GB per model (one model per fold, per method). Put
  `results_root` on a disk with room.
- **Running a model on its own** still works exactly as before, writing to that fork's own
  `aircheck/output/<experiment>/`. The pipeline's `--root` keeps its runs separate from those.
- **Metrics are not identical across machines.** Embeddings computed on a GPU and on a CPU differ by about
  1e-6, which can move hit@50 by a molecule or two. Compare models within one experiment, which is what the
  report does; `run.json` records each run's hardware, fork commit and core version.
