#!/usr/bin/env python3
"""
Enhanced Evolutionary Genetics Pipeline (v6.0)
Distinguishing Introgression and Incomplete Lineage Sorting
Using Branch Lengths - Memory Optimized, Scenario-Based
"""

import os
import time
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from Bio import Phylo
import tskit
import msprime
from concurrent.futures import ProcessPoolExecutor
from functools import wraps
import psutil
import errno
import signal
from typing import Dict, List, Tuple, Optional, Union

# Configure logging and styling
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("simulation.log"),
                        logging.StreamHandler()
                    ])
sns.set_theme(style="whitegrid", palette="viridis")
pd.set_option('display.precision', 3)

# Memory management and monitoring
def get_memory_usage():
    """Monitor memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def log_memory_usage(label=""):
    """Log current memory usage with custom label"""
    mem = get_memory_usage()
    logging.info(f"Memory usage {label}: {mem:.2f} MB")

class TimeoutError(Exception):
    """Custom exception for simulation timeouts"""
    pass

def timeout(seconds=120):
    """Function decorator to timeout long-running simulations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def _handle_timeout(signum, frame):
                raise TimeoutError(f"Simulation timed out after {seconds} seconds")
            
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator

# Configuration for scenarios - expanded parameter ranges
SCENARIOS = {
    "base": {
        "description": "Standard model with typical speciation parameters",
        "base_pop_size": 1000,
        "mutation_rate_range": (1e-8, 1e-7),
        "recombination_rate_range": (1e-9, 1e-7),  # Expanded upper bound
        "introgression_time_range": (500, 2500),
        "introgression_proportion_range": (0.05, 0.2),
        "migration_period": None
    },
    "rapid_radiation": {
        "description": "Rapid speciation events with population expansions",
        "base_pop_size": 1000,
        "expansion_size": 2000,  # Population expands during radiation
        "expansion_time": 1500,
        "mutation_rate_range": (1e-8, 1e-7),
        "recombination_rate_range": (1e-9, 1e-7),
        "introgression_time_range": (300, 1500),  # More recent introgression
        "introgression_proportion_range": (0.08, 0.25),  # Higher gene flow
        "migration_period": None
    },
    "bottleneck": {
        "description": "Populations experience severe bottlenecks",
        "base_pop_size": 1000,
        "bottleneck_size": 100,  # Small population during bottleneck
        "bottleneck_time": 1800,
        "mutation_rate_range": (1e-8, 1e-7),
        "recombination_rate_range": (1e-9, 1e-7),
        "introgression_time_range": (1000, 2000),
        "introgression_proportion_range": (0.05, 0.15),
        "migration_period": None
    },
    "continuous_migration": {
        "description": "Continuous low-level gene flow instead of pulse introgression",
        "base_pop_size": 1000,
        "mutation_rate_range": (1e-8, 1e-7),
        "recombination_rate_range": (1e-9, 1e-7),
        "migration_period": (500, 1500),  # Period with continuous migration
        "migration_rate_range": (1e-5, 1e-4)  # Migration rate instead of proportion
    }
}

def create_demography(scenario_name: str, params: Dict) -> msprime.Demography:
    """
    Create demographic models with proper event ordering for different scenarios
    
    Parameters:
    -----------
    scenario_name : str
        Name of the demographic scenario to model
    params : Dict
        Simulation parameters
        
    Returns:
    --------
    msprime.Demography
        Configured demographic model
    """
    scenario = SCENARIOS[scenario_name]
    demography = msprime.Demography()
    
    # Add populations with adjusted sizes
    demography.add_population(name="Species_A", initial_size=scenario.get('base_pop_size', 1000))
    demography.add_population(name="Species_B", initial_size=scenario.get('base_pop_size', 1000))
    demography.add_population(name="Outgroup", initial_size=1000)  # Reduced from 10,000
    demography.add_population(name="Ancestral_AB", initial_size=5000)
    demography.add_population(name="Ancestral_all", initial_size=5000)  # Common ancestor
    
    # Connect populations in the species tree
    demography.add_population_split(
        time=3000,  # Split Outgroup earlier than Species_A/B
        derived=["Outgroup", "Ancestral_AB"],
        ancestral="Ancestral_all"
    )
    
    demography.add_population_split(
        time=2000,
        derived=["Species_A", "Species_B"],
        ancestral="Ancestral_AB"
    )

    # Apply scenario-specific demographic events
    if scenario_name == "rapid_radiation":
        demography.add_population_parameters_change(
            time=scenario.get('expansion_time', 1500), 
            population="Species_A", 
            initial_size=scenario.get('expansion_size', 2000)
        )
        demography.add_population_parameters_change(
            time=scenario.get('expansion_time', 1500), 
            population="Species_B", 
            initial_size=scenario.get('expansion_size', 2000)
        )
    elif scenario_name == "bottleneck":
        demography.add_population_parameters_change(
            time=scenario.get('bottleneck_time', 1800), 
            population="Species_A", 
            initial_size=scenario.get('bottleneck_size', 100)
        )
        demography.add_population_parameters_change(
            time=scenario.get('bottleneck_time', 1800), 
            population="Species_B", 
            initial_size=scenario.get('bottleneck_size', 100)
        )
    elif scenario_name == "continuous_migration":
        # Start migration period
        migration_start, migration_end = scenario.get('migration_period', (500, 1500))
        migration_rate = params.get('migration_rate', 1e-5)
        
        # Add migration from A to B
        demography.add_migration_rate_change(
            time=migration_start, source="Species_A", dest="Species_B", rate=migration_rate
        )
        # End migration
        demography.add_migration_rate_change(
            time=migration_end, source="Species_A", dest="Species_B", rate=0
        )
    
    # Sort events by time (oldest first)
    demography.sort_events()

    # Validate demographic model
    for event in demography.events:
        if event.time < 0:
            raise ValueError(f"Negative event time: {event}")
    if any(pop.initial_size <= 0 for pop in demography.populations):
        raise ValueError("Invalid population size")

    return demography

@timeout(180)  # Extended timeout for complex scenarios
def run_simulation(params: Dict) -> Dict:
    """
    Parallel-safe simulation function with modern API
    
    Parameters:
    -----------
    params : Dict
        Simulation parameters
        
    Returns:
    --------
    Dict
        Results including tree sequence or error information
    """
    start_time = time.time()
    log_memory_usage(f"Before simulation {params.get('job_id', 'unknown')}")
    
    try:
        demography = create_demography(params['scenario'], params)
        
        # Handle pulse introgression or continuous migration
        if params['scenario'] != "continuous_migration":
            # Add introgression event
            demography.add_mass_migration(
                time=params['introgression_time'],
                source=0,  # Species_A
                dest=1,    # Species_B
                proportion=params['introgression_proportion']
            )
            demography.sort_events()

        # Set up recombination map
        recombination_map = msprime.RateMap.uniform(
            rate=params['recombination_rate'],
            sequence_length=params['length']
        )

        # Run ancestry simulation
        ts = msprime.sim_ancestry(
            samples=[
                msprime.SampleSet(10, population="Species_A"),
                msprime.SampleSet(10, population="Species_B"),
                msprime.SampleSet(2, population="Outgroup")
            ],
            demography=demography,
            recombination_rate=recombination_map,
            random_seed=params.get('seed', random.randint(1, 1_000_000)),
        )
        
        # Add mutations
        ts = msprime.sim_mutations(
            ts,
            rate=params['mutation_rate'],
            random_seed=params.get('seed', random.randint(1, 1_000_000)) + 1,  # Different seed
            model=msprime.BinaryMutationModel()
        )
        
        # Store run information
        run_info = {
            'job_id': params.get('job_id', 'unknown'),
            'elapsed_time': time.time() - start_time,
            'memory_usage_mb': get_memory_usage(),
            'num_trees': ts.num_trees,
            'num_sites': ts.num_sites
        }
        
        # Save a representative tree in Newick format if requested
        if params.get('save_trees', True):
            first_tree = ts.first()
            newick = first_tree.as_newick()
            run_info['first_tree_newick'] = newick
            
            # Save middle tree too for comparison
            mid_point = ts.num_trees // 2
            for i, tree in enumerate(ts.trees()):
                if i == mid_point:
                    run_info['middle_tree_newick'] = tree.as_newick()
                    break
        
        log_memory_usage(f"After simulation {params.get('job_id', 'unknown')}")
        
        return {
            'params': params,
            'ts': ts,
            'run_info': run_info,
            'error': None
        }
    except Exception as e:
        logging.error(f"Simulation failed: {str(e)}")
        return {
            'params': params,
            'ts': None,
            'run_info': {'elapsed_time': time.time() - start_time},
            'error': str(e)
        }

def calculate_dstat(ts: tskit.TreeSequence) -> float:
    """
    Modern D-statistic calculation using tskit
    
    Parameters:
    -----------
    ts : tskit.TreeSequence
        Tree sequence containing mutation information
        
    Returns:
    --------
    float
        D-statistic value
    """
    samples_A = ts.samples(population=0)
    samples_B = ts.samples(population=1)
    samples_Outgroup = ts.samples(population=2)
    
    abba = 0
    baba = 0
    genotype = ts.genotype_matrix()
    
    for site in genotype:
        freq_A = np.mean(site[samples_A])
        freq_B = np.mean(site[samples_B])
        freq_Out = np.mean(site[samples_Outgroup])
        
        # ABBA pattern: Species B shares derived allele with outgroup
        abba += (1 - freq_A) * freq_B * freq_Out
        # BABA pattern: Species A shares derived allele with outgroup
        baba += freq_A * (1 - freq_B) * freq_Out
        
    with np.errstate(invalid='ignore'):
        # Handle division by zero
        return (abba - baba) / (abba + baba) if (abba + baba) != 0 else 0

def calculate_branch_stats(ts: tskit.TreeSequence) -> Dict:
    """
    Calculate branch length statistics important for QuIBL analysis
    
    Parameters:
    -----------
    ts : tskit.TreeSequence
        Tree sequence to analyze
        
    Returns:
    --------
    Dict
        Statistics about branch lengths
    """
    internal_branches = []
    discordant_topologies = 0
    concordant_topologies = 0
    
    # Species tree has ((A,B),C) topology
    expected_topology = "(0,1),2"  # Simplified representation
    
    for tree in ts.trees():
        # Skip trees with multiple roots
        if len(tree.roots) != 1:
            continue
            
        # Get internal branch connecting A and B if they form a clade
        a_node = tree.root
        for u in tree.children(tree.root):
            leaves_under_u = set(tree.leaves(u))
            # Check if node contains only Species A and B samples
            if all(n < 20 for n in leaves_under_u):  # Assuming first 20 samples are from A and B
                concordant_topologies += 1
                # Get branch length
                internal_branches.append(tree.branch_length(u))
            else:
                discordant_topologies += 1
    
    if not internal_branches:
        return {
            'mean_internal_branch': None,
            'std_internal_branch': None,
            'min_internal_branch': None,
            'max_internal_branch': None,
            'topology_concordance': 0
        }
                
    return {
        'mean_internal_branch': np.mean(internal_branches),
        'std_internal_branch': np.std(internal_branches),
        'min_internal_branch': np.min(internal_branches),
        'max_internal_branch': np.max(internal_branches),
        'topology_concordance': concordant_topologies / (concordant_topologies + discordant_topologies) 
            if (concordant_topologies + discordant_topologies) > 0 else 0
    }

def analyze_simulation(result: Dict) -> Dict:
    """
    Analysis pipeline with error handling and extended metrics
    
    Parameters:
    -----------
    result : Dict
        Simulation result containing tree sequence
        
    Returns:
    --------
    Dict
        Analyzed metrics
    """
    try:
        if result['ts'] is None:
            return {**result['params'], 'error': 'No tree sequence'}

        ts = result['ts']
        
        # Check for valid trees
        for tree in ts.trees():
            if len(tree.roots) != 1:
                raise ValueError(f"Tree {tree.index} has {len(tree.roots)} roots")

        # Add mutation check
        if ts.num_sites == 0:
            raise ValueError("No mutations - FST undefined")
            
        # Calculate basic population genetics statistics
        pop0 = ts.samples(population=0)
        pop1 = ts.samples(population=1)
        
        fst = ts.Fst([pop0, pop1])
        d_stat = calculate_dstat(ts)
        
        # Calculate branch length statistics for QuIBL analysis
        branch_stats = calculate_branch_stats(ts)
        
        # Calculate allele frequency spectrum statistics
        afs_A = []
        afs_B = []
        for variant in ts.variants():
            # Count derived alleles in each population
            count_A = np.sum(variant.genotypes[pop0])
            count_B = np.sum(variant.genotypes[pop1])
            afs_A.append(count_A / len(pop0))
            afs_B.append(count_B / len(pop1))
            
        # Private vs shared alleles
        private_A = sum(1 for a, b in zip(afs_A, afs_B) if a > 0 and b == 0)
        private_B = sum(1 for a, b in zip(afs_A, afs_B) if a == 0 and b > 0)
        shared = sum(1 for a, b in zip(afs_A, afs_B) if a > 0 and b > 0)
        
        # Combine basic and extended metrics
        analysis = {
            **result['params'],
            'fst': fst,
            'd_stat': d_stat,
            'private_alleles_A': private_A,
            'private_alleles_B': private_B,
            'shared_alleles': shared,
            **branch_stats,
            'run_info': result['run_info'],
            'error': None
        }
        
        return analysis
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        return {**result.get('params', {}), 'error': str(e)}

def generate_params_list(config: Dict, n_sims: int) -> List[Dict]:
    """
    Generate parameter lists for different scenarios
    
    Parameters:
    -----------
    config : Dict
        Configuration settings
    n_sims : int
        Number of simulations to generate
        
    Returns:
    --------
    List[Dict]
        List of parameter dictionaries
    """
    params_list = []
    scenario_distribution = config.get('scenario_weights', {
        'base': 0.4, 
        'rapid_radiation': 0.3,
        'bottleneck': 0.2,
        'continuous_migration': 0.1
    })
    
    # Normalize weights
    total = sum(scenario_distribution.values())
    scenario_distribution = {k: v/total for k, v in scenario_distribution.items()}
    
    # Cumulative distribution for sampling
    cum_dist = {}
    cum_prob = 0
    for scenario, prob in scenario_distribution.items():
        cum_prob += prob
        cum_dist[scenario] = cum_prob
    
    # Generate parameter sets
    for i in range(n_sims):
        # Sample scenario according to weights
        rand_val = random.random()
        selected_scenario = None
        for scenario, cum_prob in cum_dist.items():
            if rand_val <= cum_prob:
                selected_scenario = scenario
                break
        
        # Get scenario parameters
        scenario_config = SCENARIOS[selected_scenario]
        
        # Sample parameter values
        if selected_scenario == "continuous_migration":
            params = {
                'job_id': f"{selected_scenario}_{i:04d}",
                'scenario': selected_scenario,
                'recombination_rate': 10**np.random.uniform(
                    np.log10(scenario_config['recombination_rate_range'][0]),
                    np.log10(scenario_config['recombination_rate_range'][1])
                ),
                'mutation_rate': 10**np.random.uniform(
                    np.log10(scenario_config['mutation_rate_range'][0]),
                    np.log10(scenario_config['mutation_rate_range'][1])
                ),
                'migration_rate': 10**np.random.uniform(
                    np.log10(scenario_config['migration_rate_range'][0]),
                    np.log10(scenario_config['migration_rate_range'][1])
                ),
                'length': config.get('genome_length', 1e5),
                'seed': random.randint(1, 1_000_000)
            }
        else:
            params = {
                'job_id': f"{selected_scenario}_{i:04d}",
                'scenario': selected_scenario,
                'recombination_rate': 10**np.random.uniform(
                    np.log10(scenario_config['recombination_rate_range'][0]),
                    np.log10(scenario_config['recombination_rate_range'][1])
                ),
                'mutation_rate': 10**np.random.uniform(
                    np.log10(scenario_config['mutation_rate_range'][0]),
                    np.log10(scenario_config['mutation_rate_range'][1])
                ),
                'introgression_time': np.random.uniform(
                    scenario_config['introgression_time_range'][0],
                    scenario_config['introgression_time_range'][1]
                ),
                'introgression_proportion': np.random.uniform(
                    scenario_config['introgression_proportion_range'][0],
                    scenario_config['introgression_proportion_range'][1]
                ),
                'length': config.get('genome_length', 1e5),
                'seed': random.randint(1, 1_000_000)
            }
        
        params_list.append(params)
    
    return params_list

def save_trees_for_visualization(results: List[Dict], output_prefix: str) -> None:
    """
    Save representative trees for visualization in Newick format
    
    Parameters:
    -----------
    results : List[Dict]
        Simulation results
    output_prefix : str
        Prefix for output files
    """
    # Create directory for trees if it doesn't exist
    os.makedirs(f"{output_prefix}_trees", exist_ok=True)
    
    # Save trees by scenario
    for i, result in enumerate(results):
        if 'run_info' in result and 'first_tree_newick' in result['run_info']:
            scenario = result['scenario']
            job_id = result['job_id']
            
            # Write first tree
            with open(f"{output_prefix}_trees/{job_id}_first.nwk", "w") as f:
                f.write(result['run_info']['first_tree_newick'])
            
            # Write middle tree if available
            if 'middle_tree_newick' in result['run_info']:
                with open(f"{output_prefix}_trees/{job_id}_middle.nwk", "w") as f:
                    f.write(result['run_info']['middle_tree_newick'])

def main(config_file: str = None) -> Dict:
    """
    Main execution pipeline with configuration options
    
    Parameters:
    -----------
    config_file : str
        Path to JSON configuration file
        
    Returns:
    --------
    Dict
        Results summary
    """
    # Load configuration
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'num_simulations': 200,
            'genome_length': 1e5,
            'max_workers': min(os.cpu_count(), 8),
            'output_prefix': 'ils_introgression',
            'save_trees': True,
            'scenario_weights': {
                'base': 0.4,
                'rapid_radiation': 0.3,
                'bottleneck': 0.2,
                'continuous_migration': 0.1
            }
        }
    
    start_time = time.time()
    logging.info(f"Starting simulation pipeline with {config['num_simulations']} simulations")
    log_memory_usage("Pipeline start")
    
    # Generate parameter sets
    params_list = generate_params_list(config, config['num_simulations'])
    
    # Setup parallel execution
    with ProcessPoolExecutor(max_workers=config['max_workers']) as executor:
        # Run simulations
        logging.info(f"Running simulations with {config['max_workers']} workers")
        results = list(executor.map(run_simulation, params_list))
        
        # Analyze results
        logging.info("Analyzing simulation results")
        analyzed = list(executor.map(analyze_simulation, results))
    
    # Filter out errors
    valid_results = [a for a in analyzed if a['error'] is None]
    error_count = len(analyzed) - len(valid_results)
    
    if valid_results:
        # Convert to DataFrame for analysis
        df = pd.DataFrame(valid_results)
        
        # Save raw results as CSV
        output_file = f"{config['output_prefix']}_results.csv"
        df.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")
        
        # Save Trees for visualization if requested
        if config.get('save_trees', True):
            save_trees_for_visualization(valid_results, config['output_prefix'])
            logging.info(f"Trees saved to {config['output_prefix']}_trees directory")
            
        # Generate basic summary statistics
        summary = {
            'total_simulations': len(analyzed),
            'successful_simulations': len(valid_results),
            'error_rate': error_count / len(analyzed) if analyzed else 0,
            'elapsed_time': time.time() - start_time,
            'scenarios': df['scenario'].value_counts().to_dict(),
            'avg_fst': df['fst'].mean(),
            'avg_d_stat': df['d_stat'].mean()
        }
        
        # Save summary
        with open(f"{config['output_prefix']}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        logging.info(f"Successfully analyzed {len(valid_results)} simulations")
        log_memory_usage("Pipeline end")
        
        return summary
    else:
        logging.error("No valid simulations to analyze")
        return {'error': 'All simulations failed', 'count': len(analyzed)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Evolutionary Genetics Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    args = parser.parse_args()
    
    main(args.config)