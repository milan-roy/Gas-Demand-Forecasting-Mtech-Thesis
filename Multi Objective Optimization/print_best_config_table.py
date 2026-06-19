import json
import glob
import os
"""
For each of the 12 ablation configs (combinations of input features,
scaling, and calendar encoding), reads the corresponding result file
in `results-weighted/` (`Results_<config_name>.txt`, JSON format keyed
by rank), selects the best-performing rank per config using a weighted
composite score:

    composite_score = 0.5 * MAPE_norm + 0.3 * MUPE_norm + 0.2 * MOPE_norm

(MAPE, MOPE = over_mean, MUPE = abs(under_mean) are min-max normalized
per-config before weighting; lower composite_score is better.
Under-prediction is weighted higher than over-prediction since it is
operationally costlier for gas procurement.)

The best rank's MAPE, MOPE, and MUPE (mean ± std) for each config is
then printed as a formatted ASCII table, with rows labeled by readable
config names (Inputs / Scaling / Calendar Encoding) and ordered to match
the thesis figure layout: Sales only before Sales + Crude, No before Yes
scaling, None before Numbers before Sin/Cos calendar encoding.

Usage:
    Place this script alongside a `results-weighted/` directory containing
    `Results_<config_name>.txt` files (JSON, each key a rank with
    mape_mean, mape_std, over_mean, over_std, under_mean, under_std),
    then run:

        python print_best_config_table.py

    Output is a formatted table printed to stdout (intended to be copied
    directly into the thesis results section).
"""

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results-weighted")

CONFIG_DISPLAY = {
    "sales_only_no_scaled_no_calender":          ("Sales only",    "No",  "None"),
    "sales_only_no_scaled_calender_numbers":      ("Sales only",    "No",  "Numbers"),
    "sales_only_no_scaled_calender_sincos":       ("Sales only",    "No",  "Sin/Cos"),
    "sales_only_scaled_no_calender":             ("Sales only",    "Yes", "None"),
    "sales_only_scaled_calender_numbers":         ("Sales only",    "Yes", "Numbers"),
    "sales_only_scaled_calender_sincos":          ("Sales only",    "Yes", "Sin/Cos"),
    "sales_and_crude_no_scaled_no_calender":      ("Sales + Crude", "No",  "None"),
    "sales_and_crude_no_scaled_calender_numbers": ("Sales + Crude", "No",  "Numbers"),
    "sales_and_crude_no_scaled_calender_sincos":  ("Sales + Crude", "No",  "Sin/Cos"),
    "sales_and_crude_scaled_no_calender":         ("Sales + Crude", "Yes", "None"),
    "sales_and_crude_scaled_calender_numbers":    ("Sales + Crude", "Yes", "Numbers"),
    "sales_and_crude_scaled_calender_sincos":     ("Sales + Crude", "Yes", "Sin/Cos"),
}

rows = []

for txt_path in sorted(glob.glob(os.path.join(RESULTS_DIR, "Results_*.txt"))):
    config_name = os.path.basename(txt_path).replace("Results_", "").replace(".txt", "")

    with open(txt_path) as f:
        data = json.load(f)

    ranks = list(data.keys())
    mape  = [data[r]["mape_mean"]          for r in ranks]
    mope  = [data[r]["over_mean"]          for r in ranks]
    mupe  = [abs(data[r]["under_mean"])    for r in ranks]   # abs: under_mean is negative

    def minmax(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn) if mx != mn else 0.0 for v in vals]

    mape_n = minmax(mape)
    mope_n = minmax(mope)
    mupe_n = minmax(mupe)

    scores = [0.5 * a + 0.2 * o + 0.3 * u for a, o, u in zip(mape_n, mope_n, mupe_n)]
    best_i = scores.index(min(scores))
    best_r = ranks[best_i]

    inputs, scaling, cal = CONFIG_DISPLAY.get(config_name, (config_name, "?", "?"))
    rows.append({
        "inputs":   inputs,
        "scaling":  scaling,
        "calendar": cal,
        "mape_mean": data[best_r]["mape_mean"],
        "mape_std":  data[best_r]["mape_std"],
        "mope_mean": data[best_r]["over_mean"],
        "mope_std":  data[best_r]["over_std"],
        "mupe_mean": data[best_r]["under_mean"],
        "mupe_std":  data[best_r]["under_std"],
        "best_rank": best_r,
        "score":     scores[best_i],
    })

# Sort to match the image order: Sales only first, then Sales + Crude; within each: No/Yes scaling grouped by calendar
INPUTS_ORDER = ["Sales only", "Sales + Crude"]
SCALING_ORDER = ["No", "Yes"]
CAL_ORDER = ["None", "Numbers", "Sin/Cos"]
rows.sort(key=lambda r: (
    INPUTS_ORDER.index(r["inputs"]),
    SCALING_ORDER.index(r["scaling"]),
    CAL_ORDER.index(r["calendar"]),
))

# Print table
col_w = [14, 9, 10, 20, 24, 24]
headers = ["Inputs", "Scaling", "Calendar\nEncoding", "MAPE (mean ± std)", "Over-Prediction\nError (mean ± std)", "Under-Prediction\nError (mean ± std)"]

def fmt_cell(mean, std):
    return f"{mean:.3f} ± {std:.3f}"

# Header
sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
print(sep)
h1 = ["Inputs", "Scaling", "Calendar", "MAPE (mean", "Over-Prediction", "Under-Prediction"]
h2 = ["",       "",        "Encoding", "± std)",  "Error (mean ± std)", "Error (mean ± std)"]
for h in (h1, h2):
    line = "|" + "|".join(f" {h[i]:<{col_w[i]}} " for i in range(len(col_w))) + "|"
    print(line)
print(sep)

for r in rows:
    mape_s = fmt_cell(r["mape_mean"], r["mape_std"])
    mope_s = fmt_cell(r["mope_mean"], r["mope_std"])
    mupe_s = fmt_cell(r["mupe_mean"], r["mupe_std"])
    line = (f"| {r['inputs']:<{col_w[0]}} "
            f"| {r['scaling']:<{col_w[1]}} "
            f"| {r['calendar']:<{col_w[2]}} "
            f"| {mape_s:<{col_w[3]}} "
            f"| {mope_s:<{col_w[4]}} "
            f"| {mupe_s:<{col_w[5]}} |")
    print(line)
    print(sep)
