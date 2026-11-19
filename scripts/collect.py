"""Write the experiment's report, plus a short table comparing the models.

The report itself is aircheck-benchmark-core's (`aircheck-bench report`): every run folder with a run.json
under --results, scored the same way. This adds `<report>_by_model.csv`: the best run of each model on the
test set, which is the line the benchmark exists to produce.

    python scripts/collect.py --results <experiment folder> --out <experiment folder>/report.md \
        --folds <folds .npz> --test <test file>
"""

import argparse
from pathlib import Path

import pandas as pd

from aircheck_benchmark_core.report import make_report

RANK_BY = "hit@200"  # the metric that decides which run of a model is its best
SHOW = ["hit@200", "hit@100", "hit@50", "rocauc", "pr_auc"]


def best_per_model(summary: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """One row per model: its best run, by RANK_BY.

    A run made with --no-final has no `test` split, only `test_ensemble`, so each model falls back on its
    own. `.head(1)` after sorting keeps one real row; `GroupBy.first()` would take each column's first
    non-empty value and could mix two runs into a row that never existed.
    """
    rows = summary[summary["split"].isin([split, "test_ensemble"])].copy()
    if rows.empty:
        return pd.DataFrame()
    # Prefer the asked-for split where a model has it
    rows["preferred"] = (rows["split"] == split).astype(int)
    best = (rows.sort_values(["preferred", RANK_BY], ascending=False)
                .groupby("model", as_index=False, sort=False).head(1))
    columns = [c for c in ["model", "name", "method", "split", "n", "n_pos", *SHOW] if c in best.columns]
    return best[columns].sort_values(RANK_BY, ascending=False).reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="the experiment folder, searched for run.json files")
    parser.add_argument("--out", required=True, help="where to write report.md")
    parser.add_argument("--folds", help="folds .npz: check every run used exactly these folds")
    parser.add_argument("--test", help="test file, for the report's Data section")
    parser.add_argument("--title")
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--label-col", default="LABEL")
    args = parser.parse_args(argv)

    folds = Path(args.folds) if args.folds else None
    train = str(folds).replace("_folds.npz", "_data.csv") if folds else None
    paths = make_report(args.results, out=args.out, train_path=train, test_path=args.test, folds_npz=folds,
                        title=args.title, smiles_col=args.smiles_col, label_col=args.label_col)

    summary = pd.read_csv(paths["summary"])
    best = best_per_model(summary)
    out = Path(args.out)
    by_model = out.with_name(f"{out.stem}_by_model.csv")
    if best.empty:
        print("No test predictions found, so no comparison between models was written.")
    else:
        best.to_csv(by_model, index=False)
        print(f"Best run per model, by {RANK_BY} on the test set:")
        print(best.round(4).to_string(index=False))
        print(f"Saved {by_model}")
    for path in paths.values():
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
