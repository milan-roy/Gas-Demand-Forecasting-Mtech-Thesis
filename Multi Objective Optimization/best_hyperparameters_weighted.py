import pandas as pd
import glob
import os
import json

"""
Selects the top-5 hyperparameter configurations from each Pareto-front CSV
(`pareto_front-*.csv`) found in the "pareto-front" subdirectory next to this
script, ranked by a weighted composite error score:

    composite_score = 0.5 * MAPE_norm + 0.3 * MUPE_norm + 0.2 * MOPE_norm

(MAPE, UnderError, OverError are min-max normalized per-config before
weighting; lower composite_score is better. Under-prediction is weighted
higher than over-prediction since it is operationally costlier for gas
procurement.)

For each config (FFNN/GRU/Transformer/etc., inferred from the filename),
the top 5 trials are written as a ranked dict of hyperparameters
(num_layers, hidden_layers, dropout, lr, batch_size, under_parameter,
over_parameter) to `best_hyperparameters_weighted.txt`, as a
JSON-formatted `hyperparameters = {...}` variable.

Usage:
    Place this script in the parent directory of a "pareto-front" folder
    containing one or more `pareto_front-<config_name>.csv` files (must
    contain columns: MAPE, UnderError, OverError, num_layers,
    layer_size_1..N, dropout, lr, batch_size, under_parameter,
    over_parameter), then run:

        python best_hyperparameters_weighted.py

    Output is written to pareto-front/best_hyperparameters_weighted.txt.
"""

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pareto-front")
TOP_N = 5

results = {}

for csv_path in sorted(glob.glob(os.path.join(CSV_DIR, "pareto_front-*.csv"))):
    config_name = os.path.basename(csv_path).replace("pareto_front-", "").replace(".csv", "")
    df = pd.read_csv(csv_path)

    for col, norm_col in [("MAPE", "MAPE_norm"), ("UnderError", "MUPE_norm"), ("OverError", "MOPE_norm")]:
        mn, mx = df[col].min(), df[col].max()
        df[norm_col] = (df[col] - mn) / (mx - mn) if mx != mn else 0.0

    df["composite_score"] = 0.5 * df["MAPE_norm"] + 0.3 * df["MUPE_norm"] + 0.2 * df["MOPE_norm"]
    top = df.nsmallest(TOP_N, "composite_score").reset_index(drop=True)

    config_dict = {}
    for rank_idx, row in top.iterrows():
        num_layers = int(row["num_layers"])
        hidden_layers = [
            int(row[f"layer_size_{i}"])
            for i in range(1, num_layers + 1)
        ]
        entry = {
            "rank": rank_idx + 1,
            "composite_score": round(float(row["composite_score"]), 6),
            "MAPE": float(row["MAPE"]),
            "MUPE": float(row["UnderError"]),
            "MOPE": float(row["OverError"]),
            "num_layers": num_layers,
            "hidden_layers": hidden_layers,
            "dropout": float(row["dropout"]),
            "lr": float(row["lr"]),
            "batch_size": int(row["batch_size"]),
            "under_parameter": float(row["under_parameter"]),
            "over_parameter": float(row["over_parameter"]),
        }
        config_dict[f"rank_{rank_idx + 1}"] = entry

    results[config_name] = config_dict

output_path = os.path.join(CSV_DIR, "best_hyperparameters_weighted.txt")
with open(output_path, "w") as f:
    f.write("# Best 5 hyperparameters per config (composite score = 0.5*MAPE_norm + 0.3*MUPE_norm + 0.2*MOPE_norm, lower is better)\n\n")
    f.write("hyperparameters = ")
    f.write(json.dumps(results, indent=4))
    f.write("\n")

print(f"Written to {output_path}")
