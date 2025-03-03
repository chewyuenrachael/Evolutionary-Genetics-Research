## **README for `ils_introgression_simulation.py`**

# Incomplete Lineage Sorting and Introgression Simulation

### **Overview**

This script simulates gene genealogies under a coalescent model incorporating **Incomplete Lineage Sorting (ILS)** and **introgression events**. It allows you to specify parameters such as effective population size, sample sizes per species, species split times, introgression events, and mutation rates. The output includes a printed coalescent tree and the calculation of the Site Frequency Spectrum (SFS), which is essential for analyzing genetic variation patterns in evolutionary genetics.

### **Features**

- **Coalescent Simulation**: Models the ancestral relationships among gene copies within and between species.
- **Incomplete Lineage Sorting (ILS)**: Simulates scenarios where ancestral polymorphisms are retained across species.
- **Introgression Events**: Allows for the incorporation of gene flow between species at specified times.
- **Mutation Assignment**: Assigns mutations along branches based on a mutation rate, enabling the study of genetic diversity.
- **Site Frequency Spectrum (SFS) Calculation**: Computes the distribution of allele frequencies across the sample.
- **Tree Printing**: Outputs the structure of the coalescent tree for visualization and analysis.

### **Requirements**

- **Python 3.x**
- **NumPy** library

### **Installation**

1. **Install Python 3.x**:

   Download and install from the [official Python website](https://www.python.org/downloads/).

2. **Install NumPy**:

   Open a terminal or command prompt and run:

   ```bash
   pip install numpy
   ```

3. **Download the Script**:

   Save the `ils_introgression_simulation.py` file to your working directory.

### **Usage**

1. **Set Simulation Parameters**:

   Open the script in a text editor or IDE and modify the parameters in the `main()` function to suit your research needs.

   ```python
   # Simulation parameters
   N_e = 10000  # Effective population size
   samples_per_species = {'Species_A': 5, 'Species_B': 5}  # Samples per species
   species_split_times = {'Species_A': 2000.0}  # Speciation time for Species_A
   introgression_events = [
       {'time': 1500.0, 'donor_species': 'Species_B', 'recipient_species': 'Species_A'}
   ]  # Introgression events
   mutation_rate = 1e-8  # Mutation rate per generation
   ```

   - **Effective Population Size (`N_e`)**: Set the diploid effective population size.
   - **Samples per Species (`samples_per_species`)**: Dictionary specifying the number of samples from each species.
   - **Species Split Times (`species_split_times`)**: Dictionary with species as keys and their split times as values.
   - **Introgression Events (`introgression_events`)**: List of dictionaries specifying introgression events, each with `time`, `donor_species`, and `recipient_species`.
   - **Mutation Rate (`mutation_rate`)**: Mutation rate per generation.

2. **Run the Simulation**:

   Navigate to the directory containing the script and execute:

   ```bash
   python ils_introgression_simulation.py
   ```

3. **Analyze the Output**:

   - **Coalescent Tree**: The script prints the tree structure, displaying nodes with labels, ages, and species information.
   - **Site Frequency Spectrum (SFS)**: Provides the frequencies of mutations occurring at different allele frequencies within your sample.

### **Example Output**

```
Coalescent Tree:
Node(label=Node_3500.00, age=3500.00, species=Species_A_descendant)
    Node(label=Node_3000.00, age=3000.00, species=Species_A_descendant)
        Node(label=Node_2500.00, age=2500.00, species=Species_A_descendant)
            Node(label=Species_A_0, age=0.00, species=Species_A_descendant)
            Node(label=Species_A_1, age=0.00, species=Species_A_descendant)
        Node(label=Species_A_2, age=0.00, species=Species_A_descendant)
    Node(label=Node_3200.00, age=3200.00, species=Species_B)
        Node(label=Species_B_0, age=0.00, species=Species_B)
        Node(label=Species_B_1, age=0.00, species=Species_B)

Site Frequency Spectrum (SFS):
Frequency 2/10: 3 mutations
Frequency 3/10: 2 mutations
Frequency 5/10: 1 mutations
```

### **Functions Explained**

- **`Node` Class**:

  Represents a node in the genealogy tree, with attributes for left and right child nodes, age, species, label, and mutations.

- **`simulate_coalescent_with_introgression()`**:

  Simulates the coalescent process, including ILS and introgression events, based on the specified parameters.

- **`assign_mutations()`**:

  Recursively assigns mutations to each branch of the tree according to the mutation rate.

- **`collect_site_frequency_spectrum()`**:

  Calculates the Site Frequency Spectrum (SFS) by determining the frequencies of mutations across the sample.

- **`print_tree()`**:

  Recursively prints the tree structure to visualize the genealogical relationships.

- **`main()`**:

  The main function where the simulation parameters are defined, and the simulation is executed.

### **Customizing the Simulation**

- **Adding More Species**:

  Include additional species in `samples_per_species` and define their speciation times in `species_split_times`.

  ```python
  samples_per_species = {'Species_A': 5, 'Species_B': 5, 'Species_C': 5}
  species_split_times = {'Species_A': 2000.0, 'Species_B': 2500.0}
  ```

- **Defining Multiple Introgression Events**:

  Extend the `introgression_events` list with more events.

  ```python
  introgression_events = [
      {'time': 1500.0, 'donor_species': 'Species_B', 'recipient_species': 'Species_A'},
      {'time': 1800.0, 'donor_species': 'Species_C', 'recipient_species': 'Species_B'}
  ]
  ```

- **Adjusting Mutation Rate**:

  Modify `mutation_rate` to simulate different levels of genetic variation.

### **Notes**

- **Randomness**:

  The simulation relies on random processes. For reproducible results, set a seed for the random number generator:

  ```python
  rng = np.random.default_rng(seed=42)
  ```
