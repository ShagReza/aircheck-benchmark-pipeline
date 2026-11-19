"""Run every model of the AIRCHECK benchmark on the same data and write one report.

    snakemake -n                      # show the plan, run nothing
    snakemake --cores 4               # run it
    snakemake --cores 4 --resources gpu=1   # one GPU model at a time
    snakemake check --cores 1         # only check that the configured environments work

Three steps per experiment, in <results_root>/<experiment>/:

    folds    aircheck-bench make-folds          one set of folds, shared by every model
    run      <fork>/aircheck/benchmark/run_all.sh    one job per model, in that model's environment
    report   aircheck-bench report               one report over every run in the folder

The pipeline trains nothing itself: each fork's run_all.sh does the work, and skips runs it has already
finished, so an interrupted pipeline carries on where it stopped.

Settings are in config/: config.yaml (what to run), models.yaml (the forks), datasets.yaml (the data).
"""

import re
from pathlib import Path


configfile: "config/config.yaml"


MODELS = {}
DATASETS = {}


def _load(path: str) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text()) or {}


MODELS = _load("config/models.yaml")
DATASETS = _load("config/datasets.yaml")

ROOT = Path(config["results_root"]).expanduser()
EXPERIMENTS = config["experiments"]
WANTED = config.get("models") or list(MODELS)
unknown = [m for m in WANTED if m not in MODELS]
if unknown:
    raise ValueError(f"config.yaml lists model(s) {unknown}, which are not in config/models.yaml")


def path_of(value) -> Path:
    return Path(str(value)).expanduser()


def dataset_of(experiment: str) -> dict:
    name = EXPERIMENTS[experiment]["dataset"]
    if name not in DATASETS:
        raise ValueError(f"Experiment {experiment!r} uses dataset {name!r}, which is not in config/datasets.yaml")
    return DATASETS[name]


def folds_npz(experiment: str) -> str:
    """Where `aircheck-bench make-folds` will put the fold indices for this experiment."""
    data = dataset_of(experiment)
    stem = path_of(data["train"]).stem
    method = EXPERIMENTS[experiment]["cv_method"]
    return str(ROOT / experiment / "folds" / f"{stem}_{method}_cv_folds.npz")


def model_done(experiment: str, model: str) -> str:
    """Marker a model's runs are finished; the runs themselves are folders we do not list up front."""
    return str(ROOT / experiment / ".pipeline" / f"{model}.done")


def folds_made(experiment: str) -> str:
    """Marker the folds exist. The fold file's name depends on the train file, which Snakemake cannot put in
    an output pattern, so the rule declares this marker and the path is passed on as a parameter."""
    return str(ROOT / experiment / "folds" / ".made")


# Only the names in the config are valid wildcards, so a typo fails clearly instead of matching anything
wildcard_constraints:
    experiment="|".join(re.escape(name) for name in EXPERIMENTS),
    model="|".join(re.escape(name) for name in MODELS),


rule all:
    input:
        [str(ROOT / experiment / "report.md") for experiment in EXPERIMENTS],


rule check:
    """Confirm every configured environment exists and can import its model, before a long run."""
    run:
        for name in WANTED:
            spec = MODELS[name]
            env, repo = path_of(spec["env"]), path_of(spec["repo"])
            recipe = repo / "aircheck" / "benchmark" / "run_all.sh"
            if not (env / "bin" / "python").exists():
                raise FileNotFoundError(f"{name}: no environment at {env} (see {repo}/aircheck/INSTALL.md)")
            if not recipe.exists():
                raise FileNotFoundError(f"{name}: no benchmark recipe at {recipe}")
            shell(f"{env}/bin/python -c \"import {spec['import_check']}; "
                  f"import aircheck_benchmark_core as c; print('{name}: ok, core', c.__version__)\"")


rule folds:
    """The cross-validation folds, made once and read by every model."""
    output:
        made=touch(str(ROOT / "{experiment}" / "folds" / ".made")),
    params:
        npz=lambda w: folds_npz(w.experiment),
        train=lambda w: str(path_of(dataset_of(w.experiment)["train"])),
        out=lambda w: str(ROOT / w.experiment / "folds"),
        method=lambda w: EXPERIMENTS[w.experiment]["cv_method"],
        n_folds=lambda w: ("--n-folds " + str(EXPERIMENTS[w.experiment]["n_folds"])
                           if EXPERIMENTS[w.experiment].get("n_folds") else ""),
        smiles=lambda w: dataset_of(w.experiment)["smiles_col"],
        label=lambda w: dataset_of(w.experiment)["label_col"],
        fingerprint=lambda w: dataset_of(w.experiment).get("fingerprint_col", "ECFP4"),
    conda:
        "envs/pipeline.yaml"
    shell:
        "aircheck-bench make-folds --train {params.train} --out {params.out} --method {params.method} "
        "--smiles-col {params.smiles} --label-col {params.label} --fingerprint-col {params.fingerprint} "
        "{params.n_folds} && test -f {params.npz}"


rule run_model:
    """One model's runs, in that model's own environment. The fork decides what its runs are."""
    input:
        folds=lambda w: folds_made(w.experiment),
    output:
        done=touch(str(ROOT / "{experiment}" / ".pipeline" / "{model}.done")),
    log:
        str(ROOT / "{experiment}" / ".pipeline" / "{model}.log"),
    params:
        recipe=lambda w: str(path_of(MODELS[w.model]["repo"]) / "aircheck" / "benchmark" / "run_all.sh"),
        env=lambda w: str(path_of(MODELS[w.model]["env"])),
        root=str(ROOT),
        train=lambda w: str(path_of(dataset_of(w.experiment)["train"])),
        test=lambda w: str(path_of(dataset_of(w.experiment)["test"])),
        smiles=lambda w: dataset_of(w.experiment)["smiles_col"],
        label=lambda w: dataset_of(w.experiment)["label_col"],
        fingerprint=lambda w: dataset_of(w.experiment).get("fingerprint_col", "ECFP4"),
        method=lambda w: EXPERIMENTS[w.experiment]["cv_method"],
        folds=lambda w: folds_npz(w.experiment),
        only=lambda w: ("--only " + ",".join(MODELS[w.model]["only"]) if MODELS[w.model].get("only") else ""),
        extra=lambda w: " ".join(MODELS[w.model].get("extra_args", [])),
    resources:
        gpu=lambda wildcards, attempt: int(MODELS[wildcards.model].get("gpu", 0)),
    shell:
        # The fork's recipe writes into <root>/<experiment>/; --no-report because this pipeline writes one
        # report over every model at the end
        "PATH={params.env}/bin:$PATH {params.recipe} "
        "--experiment {wildcards.experiment} --root {params.root} "
        "--folds {params.folds} --train {params.train} --test {params.test} "
        "--smiles-col {params.smiles} --label-col {params.label} "
        "--fingerprint-col {params.fingerprint} --cv-method {params.method} --no-report "
        "{params.only} {params.extra} > {log} 2>&1"


rule report:
    """One report over every model's runs in the experiment, scored by aircheck-benchmark-core."""
    input:
        models=lambda w: [model_done(w.experiment, model) for model in WANTED],
        folds=lambda w: folds_made(w.experiment),
    output:
        report=str(ROOT / "{experiment}" / "report.md"),
    params:
        results=lambda w: str(ROOT / w.experiment),
        folds=lambda w: folds_npz(w.experiment),
        test=lambda w: str(path_of(dataset_of(w.experiment)["test"])),
        smiles=lambda w: dataset_of(w.experiment)["smiles_col"],
        label=lambda w: dataset_of(w.experiment)["label_col"],
    conda:
        "envs/pipeline.yaml"
    shell:
        "python scripts/collect.py --results {params.results} --out {output.report} "
        "--folds {params.folds} --test {params.test} "
        "--smiles-col {params.smiles} --label-col {params.label} --title {wildcards.experiment}"
