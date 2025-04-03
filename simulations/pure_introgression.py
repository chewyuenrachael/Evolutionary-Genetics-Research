import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from common_utils import (
    calculate_d_statistic,
    calculate_fst,
    save_dataframe,
    plot_kde_distribution,
    simulate_introgression_scenario
)

# ----------------------
# Configuration
# ----------------------
n_replicates = 50
num_samples = 10
seq_length = 500_000
recomb_rate = 1e-8
mut_rate = 1e-8
Ne = 10_000

div_time_abc = 1_000_000  # ancient split to minimize ILS
div_time_ab = 500_000

migration_rates = {
    "no_mig": 0,
    "low_mig": 1e-6,
    "high_mig": 1e-2
}

migration_timings = {
    "recent": 25_000,
    "intermediate": 250_000,
    "ancient": 450_000
}

mig_duration = 20_000

# ----------------------
# Simulations: Intensity Sweep
# ----------------------
print("Running pure introgression simulations (intensity sweep)...")
results_intensity = []

for mig_label, mig_rate in tqdm(migration_rates.items()):
    for rep in range(n_replicates):
        ts = simulate_introgression_scenario(
            Ne=Ne,
            div_time_ab=div_time_ab,
            div_time_abc=div_time_abc,
            mig_rate=mig_rate,
            mig_time=100_000,
            mig_duration=mig_duration,
            num_samples=num_samples,
            sequence_length=seq_length,
            recombination_rate=recomb_rate,
            mutation_rate=mut_rate
        )

        d_stat = calculate_d_statistic(ts)
        fst_ab, fst_bc = calculate_fst(ts)

        results_intensity.append({
            "scenario": "intensity",
            "migration_label": mig_label,
            "rep": rep,
            "D": d_stat,
            "FST_AB": fst_ab,
            "FST_BC": fst_bc
        })

# ----------------------
# Simulations: Timing Sweep
# ----------------------
print("Running pure introgression simulations (timing sweep)...")
results_timing = []

for timing_label, timing in tqdm(migration_timings.items()):
    for rep in range(n_replicates):
        ts = simulate_introgression_scenario(
            Ne=Ne,
            div_time_ab=div_time_ab,
            div_time_abc=div_time_abc,
            mig_rate=1e-2,  # fixed high rate
            mig_time=timing,
            mig_duration=mig_duration,
            num_samples=num_samples,
            sequence_length=seq_length,
            recombination_rate=recomb_rate,
            mutation_rate=mut_rate
        )

        d_stat = calculate_d_statistic(ts)
        fst_ab, fst_bc = calculate_fst(ts)

        results_timing.append({
            "scenario": "timing",
            "timing_label": timing_label,
            "rep": rep,
            "D": d_stat,
            "FST_AB": fst_ab,
            "FST_BC": fst_bc
        })

# ----------------------
# Save results to disk
# ----------------------
intensity_df = pd.DataFrame(results_intensity)
timing_df = pd.DataFrame(results_timing)

save_results(intensity_df, "results/pure_introgression_intensity.csv")
save_results(timing_df, "results/pure_introgression_timing.csv")

# ----------------------
# Visualization
# ----------------------
plot_kde_distribution(
    intensity_df,
    hue="migration_label",
    value="D",
    title="D-statistic by Migration Intensity",
    xlabel="D-statistic",
    output_file="figures/dstat_pure_introgression_intensity.png"
)

plot_kde_distribution(
    intensity_df,
    hue="migration_label",
    value="FST_BC",
    title="FST (B-C) by Migration Intensity",
    xlabel="FST",
    output_file="figures/fst_pure_introgression_intensity.png"
)

plot_kde_distribution(
    timing_df,
    hue="timing_label",
    value="D",
    title="D-statistic by Introgression Timing",
    xlabel="D-statistic",
    output_file="figures/dstat_pure_introgression_timing.png"
)

plot_kde_distribution(
    timing_df,
    hue="timing_label",
    value="FST_BC",
    title="FST (B-C) by Introgression Timing",
    xlabel="FST",
    output_file="figures/fst_pure_introgression_timing.png"
)

print("✅ Pure introgression simulation complete.")
