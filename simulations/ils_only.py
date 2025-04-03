import msprime
import numpy as np
import pandas as pd
from tqdm import tqdm
from common_utils import (
    calculate_d_statistic,
    calculate_fst,
    plot_kde_distribution,
    sliding_window_d
)


# =======================
# Parameters
# =======================
n_replicates = 50
seq_length = 500_000
num_samples_per_pop = 10

# Divergence times (T_ABC, T_AB)
divergence_scenarios = {
    "recent": [50_000, 25_000],
    "intermediate": [200_000, 100_000],
    "ancient": [800_000, 400_000]
}

ne_scenarios = {
    "small": 10_000,
    "large": 100_000
}

recombination_rate = 1e-8
mutation_rate = 1e-8

# =======================
# Simulation Function
# =======================
def simulate_ils_scenario(Ne, divergence_time):
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=Ne)
    demography.add_population(name="B", initial_size=Ne)
    demography.add_population(name="C", initial_size=Ne)
    demography.add_population(name="AB", initial_size=Ne)
    demography.add_population(name="ABC", initial_size=Ne)

    demography.add_population_split(
        time=divergence_time[1], derived=["A", "B"], ancestral="AB")
    demography.add_population_split(
        time=divergence_time[0], derived=["AB", "C"], ancestral="ABC")

    ts = msprime.sim_ancestry(
        samples={"A": num_samples_per_pop, "B": num_samples_per_pop, "C": num_samples_per_pop},
        demography=demography,
        sequence_length=seq_length,
        recombination_rate=recombination_rate,
        random_seed=np.random.randint(1, 1e6)
    )

    ts = msprime.sim_mutations(
        ts,
        rate=mutation_rate,
        random_seed=np.random.randint(1, 1e6)
    )

    return ts

# =======================
# Run Simulations
# =======================

def run_ils_simulations():
    results = []

    for div_label, div_times in divergence_scenarios.items():
        for ne_label, ne_val in ne_scenarios.items():
            for rep in tqdm(range(n_replicates), desc=f"{div_label}-{ne_label}"):
                ts = simulate_ils_scenario(Ne=ne_val, divergence_time=div_times)
                d, abba, baba = calculate_d_statistic(ts)
                fst_ab, fst_bc = calculate_fst(ts)

                results.append({
                    "divergence": div_label,
                    "ne": ne_label,
                    "replicate": rep,
                    "D": d,
                    "ABBA": abba,
                    "BABA": baba,
                    "FST_AB": fst_ab,
                    "FST_BC": fst_bc
                })

    df = pd.DataFrame(results)
    df.to_csv("results/ils_only_results.csv", index=False)
    return df

# =======================
# Visualizations
# =======================

def plot_ils_distributions(df):
    for ne_label in ne_scenarios:
        subset = df[df["ne"] == ne_label]
        color_map = {k: f"C{i}" for i, k in enumerate(divergence_scenarios)}
        plot_kde_distribution(
            data=subset,
            group_col="divergence",
            value_col="D",
            title=f"D-statistic Distribution (Ne = {ne_label})",
            xlabel="D-statistic",
            ylabel="Density",
            color_map=color_map,
            save_path=f"results/dstat_distributions/dstat_ne_{ne_label}.png"
        )

        plot_kde_distribution(
            data=subset,
            group_col="divergence",
            value_col="FST_AB",
            title=f"FST_AB Distribution (Ne = {ne_label})",
            xlabel="FST_AB",
            ylabel="Density",
            color_map=color_map,
            save_path=f"results/fst_distributions/fst_ab_ne_{ne_label}.png"
        )

# =======================
# Main Execution
# =======================

if __name__ == "__main__":
    df = run_ils_simulations()
    plot_ils_distributions(df)