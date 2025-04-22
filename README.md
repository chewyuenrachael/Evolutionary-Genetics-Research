# 🧪 Distinguishing Introgression and Incomplete Lineage Sorting using D-statistic and FST

This repository contains the full implementation of my research project. This study simulates evolutionary scenarios using coalescent models and analyzes genomic data to distinguish between two major causes of gene tree discordance: **introgression** and **incomplete lineage sorting (ILS)**. By leveraging two complementary summary statistics, the **D-statistic** and **FST**, the study proposes a robust, joint-metric framework for inference under complex evolutionary histories.

---

## 📂 Abstract

Incomplete lineage sorting (ILS) and introgression are two major sources of gene tree discordance in evolutionary genomics, yet distinguishing between them remains a central challenge. This study systematically evaluates the performance of two key population genetic statistics—the D-statistic and fixation index (FST)—in differentiating these processes under a range of realistic evolutionary scenarios. Using coalescent simulations on a four-taxon phylogeny, we vary effective population sizes, divergence times, migration rates, timing, and directionality of gene flow to generate pure ILS, pure introgression, and mixed conditions. Our results reveal that D-statistics are highly sensitive to directional gene flow but exhibit saturation at high migration rates, while FST remains robust to weak introgression but declines with increased homogenization. Crucially, under intermediate divergence and overlapping ILS-introgression regimes, neither metric alone provides reliable inference. However, their joint distribution captures the nonlinear and complementary behavior of each, substantially improving power to distinguish evolutionary histories. We propose a two-metric inference framework and identify parameter regimes where signal detection is most effective. These findings provide theoretical and practical insight for interpreting gene tree discordance in empirical systems, particularly those involving rapid radiations or ancient introgression events.

---

## 🔬 Overview of Project Design

This project explores how to distinguish between introgression and incomplete lineage sorting (ILS)—two processes that create discordant gene trees—by systematically simulating genomic data and analyzing two summary statistics: the **D-statistic** and **FST**.

The study is organized into three main stages:

1. **Simulation of Evolutionary Scenarios**  
   Using the `msprime` coalescent simulator, we generate synthetic genomic data under a range of conditions, including:
   - Pure ILS (no migration)
   - Pure Introgression (migration between non-sister taxa)
   - Mixed Scenarios (ILS + introgression)
   - Sliding windows and genome block heterogeneity

2. **Statistical Evaluation**  
   For each simulation:
   - We compute the D-statistic and FST across replicates.
   - We analyze these metrics using both global genome-wide statistics and local sliding window analyses.

3. **Joint Inference Framework**  
   The results are visualized in Jupyter notebooks, using KDE plots, power curves, and joint metric distributions. By integrating D-statistic and FST behaviors, we identify the conditions where joint analysis outperforms single-metric detection.

This structure enables fine-grained control over evolutionary parameters and results in a robust framework to evaluate metric behavior under controlled, known histories.

---

## 📂 Project Structure

```
project_root/
├── simulations/                      # Python scripts for generating simulations
│   ├── ils_only.py                   # Pure ILS simulations
│   ├── pure_introgression.py        # Pure introgression simulations
│   ├── mixed_introgression.py       # Combined ILS + introgression simulations
│   ├── parameter_sweep.py           # Migration, Ne, and timing parameter sweeps
│   ├── genome_structure.py          # Simulation of genome block heterogeneity
│   └── common_utils.py              # Shared utilities: D-stat, FST, KDE, plotting
│
├── notebooks/                       # Analysis & visualization notebooks
│   ├── 1_ILS_only_results.ipynb
│   ├── 2_Pure_introgression.ipynb
│   ├── 3_Mixed_introgression.ipynb
│   ├── 4_Power_curves.ipynb
│   ├── 5_KDE_exploratory.ipynb
│   ├── 6_Joint_Inference.ipynb
│   └── figures/                     # All printed figures from simulation notebooks
│       ├── expanded_joint/
│       ├── expanded_kde/
│       ├── expanded_power/
│       ├── final_kde_plots/
│       ├── kde_plots/
│       ├── mixed_updated/
│       ├── power/
│       ├── section_5_4/
│       └── final_insightful_plot_FIXED.png
│
├── results/                         # Final CSV outputs from simulations
│   ├── dstat_distributions/
│   │   ├── ils_only.csv
│   │   ├── mixed_introgression.csv
│   │   └── pure_introgression.csv
│   ├── fst_distributions/
│   │   ├── ils_only.csv
│   │   ├── pure_introgression.csv
│   │   └── fst_windows_bc.csv
│   └── extreme_migration_results.csv
│
├── .gitignore
└── README.md                        # Project overview and usage guide
```

---

## 🔧 How the Simulation Scripts Work

This project uses a modular simulation architecture built with Python and `msprime`, designed to simulate coalescent genealogies under a range of evolutionary conditions. The simulation scripts are located in the `simulations/` folder, and share common logic through `common_utils.py`.

### 📌 Core Libraries Used
| Library         | Role                                                                 |
|----------------|----------------------------------------------------------------------|
| `msprime`      | Coalescent simulation of genetic ancestry (tree sequences)           |
| `tskit`        | Manipulation and analysis of tree sequences                          |
| `numpy/pandas` | Numerical operations and data wrangling                              |
| `matplotlib`   | Plotting and visualization                                            |
| `scipy.stats`  | Statistical tools including KDE                                       |
| `argparse`     | Command-line interface for passing parameters to simulation scripts  |

---

### 🧬 Modular Simulation Scripts

#### `ils_only.py`
Simulates pure incomplete lineage sorting:
- No migration (`m = 0`).
- Varies effective population size (Ne) and divergence time.
- Uses `msprime.sim_ancestry()` to generate a 4-taxon tree topology.
- Calls utility functions from `common_utils.py` to calculate D-statistic and FST from simulated tree sequences.
- Exports results to `results/` in `.csv` format.

#### `pure_introgression.py`
Simulates pure directional gene flow between non-sister taxa (C → B):
- Holds Ne and divergence time constant to minimize ILS noise.
- Varies migration rates (`m = 0, 1e-6, 1e-4, 1e-2`) and timing (recent to ancient).
- Implements one-pulse migration using `msprime.MigrationRateChange()` or custom demographic events.
- Outputs D-statistic and FST distributions, saved as CSV for later plotting.

#### `mixed_introgression.py`
Combines ILS and introgression:
- Grid-sweeps over Ne, divergence time, and gene flow parameters.
- Introduces either continuous or episodic introgression on top of genealogical discordance.
- Computes both D-stat and FST for each replicate using `common_utils.py`, then stores outputs for downstream KDE and joint analysis.

#### `parameter_sweep.py`
Quantifies detection power:
- Runs multiple replicates for each parameter combination.
- Computes statistical power: fraction of replicates exceeding thresholds (e.g., |D| > 0.2).
- Results plotted into power curves and heatmaps.
- Uses shared helpers for thresholds and evaluation in `common_utils.py`.

#### `genome_structure.py`
(Used optionally) Simulates heterogeneity across genome blocks:
- Splits a 1Mb genome into variable evolutionary segments.
- Introduces block-wise introgression to mimic real genome variation.
- Useful for simulating scenarios with semi-permeable barriers to gene flow.

---

### 🧠 `common_utils.py`: Shared Functions

The `common_utils.py` module is the computational backbone for all statistical analyses in this project. It contains reusable functions for calculating summary statistics (D-statistic, FST), analyzing results across sliding windows, and exporting data. Below are detailed descriptions of the most critical functions, including their mathematical foundations and implementation insights.

- **`compute_dstat()`**: Calculates the D-statistic from simulated tree sequences.
- **`compute_fst()`**: Computes pairwise Weir & Cockerham’s FST values.
- **`sliding_window_analysis()`**: Enables local windowed statistics across the genome.
- **`save_dataframe()`**: Standardized CSV exports for results tracking.
- **`plot_distributions()`**: Reusable matplotlib-based visualizations for D, FST, and KDE plots.


---

#### 📐 `compute_dstat(ts, samples, P1, P2, P3, O)`

**Purpose**: Computes the D-statistic (ABBA-BABA test) from a `tskit.TreeSequence` object. This quantifies asymmetric allele sharing and is a key signal of introgression.

**Biological Logic**: In a four-taxon tree `(((P1, P2), P3), O)`, the D-statistic compares the frequency of two site patterns:
- **ABBA**: Derived allele is shared by P2 and P3.
- **BABA**: Derived allele is shared by P1 and P3.

Under incomplete lineage sorting alone (ILS), ABBA ≈ BABA, so **D ≈ 0**. An excess of ABBA or BABA indicates directional gene flow.

**Mathematical Definition**:
\[
D = \frac{N_{ABBA} - N_{BABA}}{N_{ABBA} + N_{BABA}}
\]

**Code Summary**:
- Loops over biallelic sites in the tree sequence.
- Uses allele frequency vectors per population (via `tskit.TreeSequence.allele_frequency_spectrum()` or similar).
- Tallies ABBA and BABA counts based on population allele configurations.
- Returns D and raw site counts.

**Robustness**: Handles multiple alleles per site (e.g., low-frequency derived alleles), masks monomorphic and tri-allelic sites, and filters for biallelic SNPs only.

---

#### 🧬 `compute_fst(ts, pop1, pop2)`

**Purpose**: Computes Wright’s FST between two populations (e.g., A-B, B-C), summarizing population differentiation.

**Biological Logic**: FST reflects how allele frequencies differ between populations. Under introgression, FST between gene-flowing populations should drop, whereas ILS maintains neutral divergence.

**Mathematical Foundation (Weir & Cockerham's estimator)**:
\[
F_{ST} = \frac{Var(p)}{p(1 - p)} = \frac{H_T - H_S}{H_T}
\]
Where:
- \( H_T \) is expected heterozygosity in the total population.
- \( H_S \) is the average heterozygosity within each subpopulation.

**Code Summary**:
- Loops through all segregating sites in the tree sequence.
- Calculates allele frequencies in pop1 and pop2.
- Computes within- and between-population heterozygosity.
- Aggregates these to produce average genome-wide FST.

**Robustness**: Avoids division-by-zero errors in monomorphic sites; supports flexible sample masks; scalable to long chromosomes or many replicates.

---

#### 🧬 `sliding_window_analysis(ts, window_size=50000, step_size=50000, stat_func=compute_dstat)`

**Purpose**: Applies a summary statistic (D-statistic or FST) in non-overlapping or sliding windows across a chromosome.

**Use Case**: Simulates real-genome heterogeneity where introgression may affect only parts of the genome. Critical for identifying local vs. global signal patterns.

**How it Works**:
1. **Defines genome windows** using start, end, and step size.
2. For each window:
   - Extracts the relevant `tskit.TreeSequence` interval.
   - Applies a provided statistical function (e.g., `compute_dstat`, `compute_fst`).
   - Collects the result into a `DataFrame` or list.
3. Returns per-window statistics with genomic coordinates.

**Configurable**:
- Works with any function of the form `func(ts, samples, ...)`.
- Can be extended to use overlapping windows, quantiles, or bootstrap resampling.


---

### 🔁 How Everything Connects

Each script simulates tree sequences using `msprime`, analyzes them using `tskit` and `common_utils.py`, and then stores the outputs for visualization in Jupyter notebooks.

| Script                  | Produces         | Used In Notebook                 |
|-------------------------|------------------|----------------------------------|
| `ils_only.py`           | Pure ILS stats   | `1_ILS_only_results.ipynb`       |
| `pure_introgression.py` | Clean introgression results | `2_Pure_introgression.ipynb`   |
| `mixed_introgression.py`| Mixed signatures | `3_Mixed_introgression.ipynb`    |
| `parameter_sweep.py`    | Power data       | `4_Power_curves.ipynb`           |

---

## 📊 Summary of Main Simulations & Results

### 1️⃣ `1_ILS_only_results.ipynb` — **Section 5.1 of Paper**
**Goal**: Establish null expectations under pure ILS with no introgression.  
**Simulation Setup**:
- Scripts used: `ils_only.py`, `common_utils.py`
- Parameters varied: 
  - Ne = 10,000 vs 100,000
  - Divergence times: AB = 25k, 100k, 400k; ABC = 50k, 200k, 800k
  - 30 replicates per setting

**Main Findings**:
| Parameter      | Effect on D-statistic           | Effect on FST                     |
|----------------|----------------------------------|-----------------------------------|
| Small Ne       | Broad distributions, extreme tails | High FST due to drift              |
| Large Ne       | Narrow D centered at 0           | Lower FST due to preserved diversity |
| Recent Divergence | Higher D variance, noisy tails     | Lower FST                          |
| Ancient Divergence | Narrow D, lower variance          | High FST                          |

This notebook forms the empirical null model against which all subsequent introgression scenarios are compared.

---

### 2️⃣ `2_Pure_introgression.ipynb` — **Section 5.2 of Paper**
**Goal**: Isolate how introgression alone (no ILS) affects D-statistic and FST.  
**Simulation Setup**:
- Script: `pure_introgression.py`
- Parameters:
  - Ne = 100,000
  - AB = 800k, ABC = 400k (to suppress ILS)
  - Directional gene flow: C → B
  - Migration rates: 0, 1e-6, 1e-4, 1e-2
  - Timing: 10k, 50k, 200k generations ago
  - 50 replicates per setting

**Key Insights**:
- D-statistic increases with migration rate (from 0 to ~0.08)
- More recent gene flow yields stronger signals
- FST BC decreases with increasing migration (homogenization)

These clean signals validate both D-statistic and FST as diagnostics for introgression under idealized conditions.

---

### 3️⃣ `3_Mixed_introgression.ipynb` — **Section 5.3 of Paper**
**Goal**: Examine real-world-like settings where ILS and introgression co-occur.
**Simulation Setup**:
- Script: `mixed_introgression.py`
- Full parameter grid:
  - Ne: 10k, 100k
  - Divergence times: recent, intermediate, ancient
  - Migration: 0, 1e-4, 1e-2
  - Timing: recent vs ancient
  - Mode: continuous vs episodic
  - 30–50 replicates per combination

**Outputs**:
- Joint D-stat vs. FST KDE plots
- Sliding window variation across 1 Mb
- KDE shifts indicate introgression regimes (lower FST + higher D)

This notebook provides the most ecologically realistic conditions. Joint metrics begin to outperform single-metric inference.

---

### 4️⃣ `4_Power_curves.ipynb` — **Section 5.4 of Paper**
**Goal**: Quantify detection sensitivity of D-statistic and FST across scenarios.
**Script**: `parameter_sweep.py`
**Power analysis**:
- Power defined as:
  - D-statistic: proportion exceeding |D| > 0.2
  - FST: proportion below null 5th percentile

**Findings**:
| Migration Rate | Power (D-statistic) | Power (FST) |
|----------------|---------------------|--------------|
| 0.0            | ~5% (false pos)     | ~5%           |
| 1e-4           | 25–40%              | 15–30%        |
| 1e-2           | >80%                | ~60–70%       |

Joint detection increases with stronger migration and large Ne. FST lags in sensitivity but complements D by being robust to ILS.

---

### 📊 Summary of Metric Behavior Across Scenarios (Sections 5.1–5.4)

| **Scenario**        | **D-statistic Behavior**                                                                 | **F<sub>ST</sub> Behavior**                                                                 |
|---------------------|-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| **Pure ILS**        | Centers near 0; no detection power                                                       | High variance; insensitive to introgression                                                  |
| **Pure Introgression** | Strong increase with gene flow; saturates at high migration                             | Steady decline with gene flow; sensitive across full range                                   |
| **Mixed Scenarios** | Skewed distributions under moderate migration; high detection power under intermediate divergence | Fails to differentiate weak introgression from ILS; limited power under intermediate divergence |
| **Joint Behavior**  | Inverse correlation with F<sub>ST</sub>; clusters reveal clear signal separation under migration | Improves interpretability when jointly analyzed with D-statistic                             |

---

## 📦 Dependencies
Install with:
```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Simulations
```bash
python simulations/ils_only.py
python simulations/pure_introgression.py
python simulations/mixed_introgression.py
python simulations/parameter_sweep.py
```

---

## 📊 Outputs
- `results/`: Raw `.csv`, `.npz`, or `.pkl` outputs
- `figures/`: Publication-ready plots (KDEs, power curves, D-FST scatter)

---

### 📄 Example Output: Summary of Simulated Replicates

Each simulation produces detailed summary statistics stored as `.csv` files in the `results/` directory. These include demographic parameters, migration settings, and computed values of D-statistic, ABBA/BABA counts, and FST.

| rep | Ne     | divergence | mig_time | mig_rate | mig_model | mig_direction | D      | ABBA     | BABA     | FST_AB  | FST_BC  |
|-----|--------|------------|----------|----------|------------|----------------|--------|----------|----------|---------|---------|
| 0   | 100000 | shallow    | recent   | none     | episodic   | C_to_B         | 0.0355 | 436.13   | 406.22   | 0.1395  | 0.1984  |
| 1   | 100000 | shallow    | recent   | none     | episodic   | C_to_B         | 0.0482 | 416.60   | 378.28   | 0.1326  | 0.1952  |
| 2   | 100000 | shallow    | recent   | none     | episodic   | C_to_B         | 0.0191 | 437.12   | 420.73   | 0.1334  | 0.2007  |

**Parameter meanings**:
- `rep`: Replicate number
- `Ne`: Effective population size
- `divergence`: Depth of species split (shallow/intermediate/deep)
- `mig_time`, `mig_rate`, `mig_model`: Temporal and intensity profile of migration
- `mig_direction`: Direction of introgression (typically C → B)
- `D`: D-statistic (ABBA-BABA imbalance)
- `ABBA` / `BABA`: Site pattern counts
- `FST_AB`, `FST_BC`: Pairwise genetic differentiation (Weir & Cockerham)

This structured output enables downstream power analysis, KDE-based clustering, and parameter-specific inference.

---

## 🧠 Scientific Contribution
- First to propose joint KDE-based D-stat & FST signatures
- Establishes practical detection thresholds (D > 0.2; FST < null range)
- Validates parameter regimes where ILS vs introgression are inferable

---

## 📜 Citation
> Rachael Chew. *Distinguishing Introgression and Incomplete Lineage Sorting in Evolutionary Genetics using D-statistic and FST*. Minerva University Capstone, 2025.

---

## 🔍 License
MIT License
