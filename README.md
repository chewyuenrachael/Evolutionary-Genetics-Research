# README: Evolutionary Genetics Analysis Pipeline

This repository contains a suite of Python scripts for simulating and analyzing evolutionary genetic data to distinguish introgression from incomplete lineage sorting (ILS) using the D-statistic and FST. The code implements multiple analysis pipelines including detection threshold analysis, statistical evaluation, visualization, and machine learning classification.

**Project title:** Distinguishing Introgression and Incomplete Lineage Sorting in Evolutionary Genetics Using the D-statistic and FST

**Abstract:** In evolutionary genomics, distinguishing introgression (gene flow between diverging lineages) from incomplete lineage sorting (ILS) is a persistent challenge. We present a comprehensive simulation study comparing the D-statistic (ABBA–BABA test) and FST in their ability to discriminate introgression from ILS. Four evolutionary scenarios were modeled—baseline divergence, continuous migration, population bottleneck, and rapid radiation—across 200 coalescent simulations. We evaluated distributional patterns of each metric, their correlation, and their performance in detecting introgression under varying conditions of gene flow, timing, population size, mutation rate, and recombination rate. Our results show that while the D-statistic and FST each have limited power alone, they capture complementary signals: FST highlights differences under high gene flow or drift, whereas D-statistics reflect asymmetrical allele sharing. Neither metric consistently exceeded a modest area-under-curve (AUC ~0.6) in distinguishing introgression from ILS, underscoring the difficulty of the task. However, combined analysis of both metrics improved discrimination, revealing distinctive joint signatures for scenarios like rapid radiation vs. continuous migration. We identify parameter thresholds where each metric performs optimally and show that FST often requires stronger gene flow or drift signals than D-statistics to detect introgression. By integrating both statistics and considering demographic context, researchers can more reliably infer whether shared genetic variation stems from introgression or ILS. These findings provide practical guidelines for evolutionary studies and highlight the importance of complementary approaches when unraveling complex speciation histories.

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Prerequisites and Dependencies](#prerequisites-and-dependencies)
4. [Installation](#installation)
5. [Usage Instructions](#usage-instructions)
6. [Configuration](#configuration)
7. [Output Files](#output-files)
8. [Troubleshooting](#troubleshooting)
9. [License](#license)

## Overview

This pipeline aims to rigorously simulate evolutionary scenarios and then analyze the resulting genetic data with various statistical and machine learning methods. The core objectives are:

- To run coalescent simulations using **msprime** and process tree sequences with **tskit**.
- To calculate summary statistics such as the D-statistic and FST.
- To perform sensitivity and threshold analyses.
- To visualize the results (distribution plots, ROC curves, scatter plots).
- To apply machine learning methods for classification (e.g., distinguishing introgression vs. ILS).

The entire suite is designed to be modular so that you can run individual components as needed or execute the entire pipeline using the master script.

## File Structure

The repository contains the following key Python files:

- **detection_threshold_analysis.py**  
  Analyzes detection thresholds for various simulation parameters.

- **dstat_fst_analysis.py**  
  Computes and compares the D-statistic and FST across simulated datasets.

- **enhanced_genetics_pipeline.py**  
  Contains advanced simulation and analysis routines to generate enhanced genetic data.

- **master_pipeline.py**  
  The main script that integrates all components and orchestrates the overall workflow from simulation through analysis and visualization.

- **ml_classification.py**  
  Implements machine learning classifiers (using scikit-learn) to distinguish between introgression and ILS scenarios.

- **run_analysis_pipeline.py**  
  A convenient script to run the entire analysis pipeline sequentially.

- **statistical_analysis.py**  
  Contains functions for statistical testing, ROC analysis, and parameter sensitivity analysis.

- **validation_methods.py**  
  Provides functions for validating simulation results and consistency checks.

- **visualization_tools.py**  
  Contains plotting routines to generate histograms, density plots, scatter plots, ROC curves, and other figures.

You may also find additional helper files and configuration files (if any) in the repository.

## Prerequisites and Dependencies

The following libraries and tools are required to run the pipeline:

- **Python 3.8+**  
- **msprime** (for coalescent simulations)  
- **tskit** (for tree sequence processing)  
- **NumPy**  
- **SciPy**  
- **Pandas**  
- **Matplotlib**  
- **Seaborn**  
- **scikit-learn** (for machine learning classification)  
- **Jupyter Notebook** (optional, for interactive exploration)

Other dependencies (if any) should be specified in the `requirements.txt` file included in the repository.

## Installation

1. **Clone the Repository**  
   Open your terminal and clone the repository:  
   ```bash
   git clone https://github.com/yourusername/evolutionary-genetics-pipeline.git
   cd evolutionary-genetics-pipeline
   ```

2. **Set Up a Virtual Environment** (optional but recommended)  
   Create and activate a virtual environment:  
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**  
   If a `requirements.txt` is provided, install with:  
   ```bash
   pip install -r requirements.txt
   ```  
   Alternatively, manually install the libraries:  
   ```bash
   pip install msprime tskit numpy scipy pandas matplotlib seaborn scikit-learn jupyter
   ```

## Usage Instructions

### Running the Entire Pipeline

The main entry point is the **master_pipeline.py** script, which integrates simulation, analysis, and visualization. To run the entire pipeline, simply execute:
```bash
python master_pipeline.py --output_dir results --num_simulations 200
```
This script will:
- Set up the simulation parameters.
- Run 200 simulation replicates for each evolutionary scenario.
- Compute the D-statistic and FST for each replicate.
- Perform statistical and ROC analyses.
- Generate figures and output summary statistics.

### Running Individual Components

You may also run individual scripts as needed:

- **Simulation and Analysis:**  
  ```bash
  python run_analysis_pipeline.py
  ```
  This script runs the simulation and statistical analysis portions.

- **Threshold and Sensitivity Analyses:**  
  ```bash
  python detection_threshold_analysis.py
  ```
  or
  ```bash
  python dstat_fst_analysis.py
  ```
  These scripts vary key parameters (introgression proportion, timing, etc.) and evaluate detection performance.

- **Machine Learning Classification:**  
  ```bash
  python ml_classification.py
  ```
  This file applies ML methods (e.g., classifiers) to the simulated data.

- **Visualization:**  
  ```bash
  python visualization_tools.py
  ```
  This script generates all the figures (density plots, ROC curves, scatter plots, etc.). Figures are saved in the `/figures` directory.

- **Validation:**  
  ```bash
  python validation_methods.py
  ```
  Use this script to run consistency and validation checks on the simulation outputs.

## Configuration

Many scripts have configuration options at the top of the file (e.g., simulation parameters, number of replicates, file paths for outputs). Check the configuration section in each file to adjust parameters for your study system. Common configuration settings include:

- **Population sizes, divergence times, mutation/recombination rates** (in simulation scripts).  
- **Number of simulation replicates** (default is 200).  
- **Migration rates and introgression proportions.**  
- **Output directories** for figures and analysis results.

Make sure that any paths you change exist or are created prior to running the scripts.

## Output Files

- **Figures:** All plots (density plots, scatter plots, ROC curves) will be saved in the `/figures` directory.  
- **Summary Statistics:** CSV or TXT files with results from statistical analyses and ML classification are saved in the `/results` directory.  
- **Logs:** A log file may be generated to record simulation details and parameter settings for reproducibility.

## Troubleshooting

- **Missing Libraries:** Ensure all required libraries are installed. Activate your virtual environment if you’re using one.  
- **Path Issues:** Verify that the output directories (e.g., `/figures` and `/results`) exist or that the scripts have permission to create them.  
- **Memory/Performance:** Some simulations (especially 200 replicates per scenario) can be memory intensive. Consider reducing the number of replicates for testing or running on a machine with sufficient resources.  
- **Error Messages:** If an error occurs, check the error message and traceback for clues about which module or function is failing. Common issues include version incompatibilities or misconfigured simulation parameters
