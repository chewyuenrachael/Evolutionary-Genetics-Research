
### **Enhancements**

- **Multiple Simulations**: Added the `run_simulations()` function to execute the simulation multiple times (`num_simulations`) and aggregate the results.
- **Aggregated SFS**: The site frequency spectrum from each simulation is collected and aggregated to calculate average mutation counts for each allele frequency.
- **Plotting Results**: Implemented the `plot_sfs_distribution()` function using `matplotlib` to visualize the average SFS across all simulations.
- **User-Controlled Parameters**: The number of simulations (`num_simulations`) can be adjusted in the `main()` function to control how many times the simulation runs.

---

# Incomplete Lineage Sorting and Introgression Simulation

### **Overview**

This script simulates gene genealogies under a coalescent model incorporating **Incomplete Lineage Sorting (ILS)** and **introgression events**. It allows you to run the simulation multiple times and aggregate the results to analyze the distribution of outcomes, providing more rigorous scientific insights.

### **Features**

- **Multiple Simulations**: Run the simulation as many times as needed to collect robust statistical data.
- **Coalescent Simulation**: Models the ancestral relationships among gene copies within and between species.
- **Incomplete Lineage Sorting (ILS)**: Simulates scenarios where ancestral polymorphisms are retained across species.
- **Introgression Events**: Incorporates gene flow between species at specified times.
- **Mutation Assignment**: Assigns mutations along branches based on a mutation rate.
- **Site Frequency Spectrum (SFS) Calculation**: Computes the distribution of allele frequencies across the sample.
- **Aggregated Results**: Collects and averages the SFS data over multiple simulations.
- **Visualization**: Plots the average SFS distribution using `matplotlib`.

### **Requirements**

- **Python 3.x**
- **NumPy** library
- **Matplotlib** library

### **Installation**

1. **Install Python 3.x**:

   Download and install from the [official Python website](https://www.python.org/downloads/).

2. **Install Required Libraries**:

   Open a terminal or command prompt and run:

   ```bash
   pip install numpy matplotlib
   ```

3. **Download the Script**:

   Save the `ils_introgression_simulation.py` file to your working directory.

### **Usage**

1. **Set Simulation Parameters**:

   Open the script in a text editor or IDE and modify the parameters in the `main()` function:

   ```python
   # Simulation parameters
   N_e = 10000  # Effective population size
   samples_per_species = {'Species_A': 5, 'Species_B': 5}  # Samples per species
   species_split_times = {'Species_A': 2000.0}  # Speciation time for Species_A
   introgression_events = [
       {'time': 1500.0, 'donor_species': 'Species_B', 'recipient_species': 'Species_A'}
   ]  # Introgression events
   mutation_rate = 1e-8  # Mutation rate per generation

   num_simulations = 100  # Number of simulations to run
   ```

   - **Effective Population Size (`N_e`)**: Set the diploid effective population size.
   - **Samples per Species (`samples_per_species`)**: Dictionary specifying the number of samples from each species.
   - **Species Split Times (`species_split_times`)**: Dictionary with species as keys and their split times as values.
   - **Introgression Events (`introgression_events`)**: List of dictionaries specifying introgression events.
   - **Mutation Rate (`mutation_rate`)**: Mutation rate per generation.
   - **Number of Simulations (`num_simulations`)**: Number of times the simulation will run.

2. **Run the Simulation**:

   Navigate to the directory containing the script and execute:

   ```bash
   python ils_introgression_simulation.py
   ```

3. **Analyze the Output**:

   - **Aggregated Site Frequency Spectrum**: The script prints the average number of mutations at each allele frequency over all simulations.
   - **Visualization**: A bar chart displays the average SFS distribution, providing a visual representation of the results.

### **Example Output**

```
Aggregated Site Frequency Spectrum (SFS) over Simulations:
Frequency 1/10: Average Mutations = 0.00
Frequency 2/10: Average Mutations = 2.35
Frequency 3/10: Average Mutations = 1.57
Frequency 4/10: Average Mutations = 0.89
Frequency 5/10: Average Mutations = 0.45
Frequency 6/10: Average Mutations = 0.22
Frequency 7/10: Average Mutations = 0.11
Frequency 8/10: Average Mutations = 0.05
Frequency 9/10: Average Mutations = 0.02
Frequency 10/10: Average Mutations = 0.01
```

A bar chart window will also pop up, showing the average SFS distribution.

### **Functions Explained**

- **`run_simulations()`**:

  Executes the simulation multiple times and aggregates the SFS results.

- **`plot_sfs_distribution()`**:

  Uses `matplotlib` to plot the average SFS over all simulations.

- **Other Functions**:

  Remain the same as previously explained, handling the simulation of the coalescent process, mutation assignment, and SFS collection.

### **Customizing the Simulation**

- **Adjusting Number of Simulations**:

  Modify `num_simulations` in the `main()` function to increase or decrease the number of simulations, which affects the statistical robustness of the results.

- **Changing Population Parameters**:

  As before, you can adjust `N_e`, `samples_per_species`, `species_split_times`, `introgression_events`, and `mutation_rate` to explore different evolutionary scenarios.

### **Notes**

- **Computation Time**:

  Increasing the number of simulations will result in longer computation times. Ensure your system can handle the load, or adjust the number accordingly.

- **Randomness**:

  For reproducible results, set a fixed seed for the random number generator:

  ```python
  rng = np.random.default_rng(seed=42)
  ```

- **Visualization**:

  The plot provides a visual summary of the average SFS, which can be useful for presentations and further analysis.

### **Applications**

- **Statistical Analysis**:

  Running multiple simulations allows for the assessment of variance and confidence intervals in your results.

- **Hypothesis Testing**:

  Compare the aggregated SFS under different scenarios (e.g., with and without introgression) to test specific evolutionary hypotheses.

- **Educational Use**:

  Demonstrate the effects of evolutionary processes on genetic diversity through interactive simulations.

### **Troubleshooting**

- **Dependencies**:

  Ensure all required libraries (`numpy`, `matplotlib`) are installed.

- **Display Issues**:

  If the plot does not display, ensure that your Python environment supports GUI operations. For remote servers or environments without display capabilities, consider saving the plot to a file by adding `plt.savefig('sfs_distribution.png')` before `plt.show()`.
