import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common_utils import (
    simulate_ancestry,
    add_mutations,
    calculate_d_statistic,
    calculate_fst,
    plot_kde
)

# -------------------------------
# Configuration
# -------------------------------

n_replicates = 30
seq_length = 5e5
num_samples_per_pop = 10
Ne = 5_0000
recomb_rate = 1e-8
mut_rate = 1e-8

output_dir = "results/mixed_introgression"
os.makedirs(output_dir, exist_ok=True)

# Parameter ranges
divergence_times = {
    'recent': [50_000, 20_000],
    'intermediate': [200_000, 80_000],
    'ancient': [500_000, 200_000],
}

migration_rates = {
    'none': 0,
    'low': 1e-6,
    'moderate': 1e-4,
    'high': 1e-2,
}

introgression_timings = {
    'recent': 10_000,
    'intermediate': 40_000,
    'ancient': 75_000,
}

# -------------------------------
# Run Simulations
# -------------------------------

all_results = []
print("[INFO] Running Mixed ILS + Introgression Simulations...")

for div_label, (t_abc, t_ab) in divergence_times.items():
    for mig_label, mig_rate in migration_rates.items():
        for timing_label, timing in introgression_timings.items():

            # Ensure introgression time precedes AB divergence
            if timing >= t_ab:
                continue

            for rep in range(n_replicates):
                ts = simulate_ancestry(
                    Ne=Ne,
                    divergence_times=(t_abc, t_ab),
                    mig_rate=mig_rate,
                    mig_start_time=timing,
                    seq_length=seq_length,
                    recomb_rate=recomb_rate,
                    num_samples=num_samples_per_pop
                )
                ts = add_mutations(ts, mut_rate)

                d_stat, abba, baba = calculate_d_statistic(ts)
                fst_ab, fst_bc = calculate_fst(ts)

                all_results.append({
                    "div_time": div_label,
                    "mig_rate": mig_label,
                    "mig_time": timing_label,
                    "rep": rep,
                    "D": d_stat,
                    "ABBA": abba,
                    "BABA": baba,
                    "FST_AB": fst_ab,
                    "FST_BC": fst_bc,
                })

# -------------------------------
# Save and Plot
# -------------------------------

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(output_dir, "mixed_introgression_results.csv"), index=False)

# Plot D-statistic by migration intensity
plot_kde(
    results_df,
    hue_col="mig_rate",
    value_col="D",
    title="D-statistic Distribution by Migration Intensity (Mixed ILS + Introgression)",
    xlabel="D-statistic",
    save_path=os.path.join(output_dir, "d_by_migration.png")
)

# Plot FST_AB by divergence time
plot_kde(
    results_df,
    hue_col="div_time",
    value_col="FST_AB",
    title="FST_AB Distribution by Divergence Time (Mixed ILS + Introgression)",
    xlabel="FST_AB",
    save_path=os.path.join(output_dir, "fst_ab_by_divtime.png")
)

# Plot D-statistic by introgression timing
plot_kde(
    results_df,
    hue_col="mig_time",
    value_col="D",
    title="D-statistic by Introgression Timing (Mixed ILS + Introgression)",
    xlabel="D-statistic",
    save_path=os.path.join(output_dir, "d_by_timing.png")
)

print("[INFO] Done. Results saved in:", output_dir)