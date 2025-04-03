import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from simulations.common_utils import (
    simulate_ancestry,
    calculate_d_statistic,
    calculate_fst,
    ensure_dir
)

# Output path setup
output_csv = "results/parameter_sweep/parameter_sweep_results.csv"
fig_dir = "results/parameter_sweep/"
ensure_dir(fig_dir)

# Constants
n_replicates = 30
seq_length = 500000
num_samples_per_pop = 10

# Default parameters
default_params = {
    "Ne": 10000,
    "divergence_time": [500000, 200000],
    "migration_rate": 1e-4,
    "migration_time": 10000,
    "migration_duration": 5000,
    "mutation_rate": 1e-8,
    "recombination_rate": 1e-8
}

# Parameter sweep ranges
sweep_params = {
    "Ne": [1000, 10000, 50000, 100000],
    "divergence_time": [[100000, 40000], [500000, 200000], [1000000, 400000]],
    "migration_rate": [0, 1e-6, 1e-4, 1e-2],
    "mutation_rate": [1e-9, 5e-9, 1e-8],
    "recombination_rate": [1e-9, 5e-9, 1e-8]
}

# Run simulations
results = []

print("Running parameter sweep...")
for param, values in sweep_params.items():
    for val in tqdm(values, desc=f"Sweeping {param}"):
        for rep in range(n_replicates):
            # Set custom parameters for this replicate
            params = default_params.copy()
            params[param] = val

            # Simulate tree sequence
            try:
                ts = simulate_ancestry(
                    Ne=params["Ne"],
                    divergence_time=params["divergence_time"],
                    migration_rate=params["migration_rate"],
                    migration_time=params["migration_time"],
                    migration_duration=params["migration_duration"],
                    mutation_rate=params["mutation_rate"],
                    recombination_rate=params["recombination_rate"],
                    seq_length=seq_length,
                    num_samples_per_pop=num_samples_per_pop
                )

                # Calculate statistics
                d_stat = calculate_d_statistic(ts)
                fst_ab, fst_bc = calculate_fst(ts)

                results.append({
                    "parameter": param,
                    "value": val,
                    "replicate": rep,
                    "D_statistic": d_stat,
                    "FST_AB": fst_ab,
                    "FST_BC": fst_bc
                })
            except Exception as e:
                print(f"Error in simulation: {e}")
                continue

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}")

# Plot D and FST vs each parameter
print("Generating plots...")
sns.set(style="whitegrid")

for param in sweep_params.keys():
    plt.figure(figsize=(10, 6))
    subset = results_df[results_df['parameter'] == param]
    sns.boxplot(x="value", y="D_statistic", data=subset)
    plt.xscale("log") if param == "migration_rate" else None
    plt.title(f"D-statistic vs {param}")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"d_stat_vs_{param}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="value", y="FST_AB", data=subset)
    plt.xscale("log") if param == "migration_rate" else None
    plt.title(f"FST (A vs B) vs {param}")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"fst_ab_vs_{param}.png"), dpi=300)
    plt.close()

print("Parameter sweep complete.")