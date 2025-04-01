#!/usr/bin/env python3
"""
Integrated Analysis Pipeline for Introgression vs ILS Research
This script orchestrates the entire workflow from simulation to visualization
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from datetime import datetime

# Configure logging
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

def run_command(cmd, description):
    """Run a command and log the output"""
    logging.info(f"Running {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{description} failed: {e}")
        logging.error(f"STDOUT: {e.stdout}")
        logging.error(f"STDERR: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the complete analysis pipeline")
    parser.add_argument("--config", help="Configuration file path (JSON)")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--skip_simulation", action="store_true", help="Skip simulation step")
    parser.add_argument("--skip_validation", action="store_true", help="Skip validation step")
    parser.add_argument("--skip_visualization", action="store_true", help="Skip visualization step")
    parser.add_argument("--n_simulations", type=int, default=200, 
                     help="Number of simulations to run")
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths to modules
    simulation_script = os.path.join(script_dir, "enhanced_genetics_pipeline.py")
    visualization_script = os.path.join(script_dir, "visualization_tools.py")
    analysis_script = os.path.join(script_dir, "statistical_analysis.py")
    validation_script = os.path.join(script_dir, "validation_methods.py")
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'num_simulations': args.n_simulations,
            'genome_length': 1e5,
            'max_workers': min(os.cpu_count(), 8),
            'output_prefix': os.path.join(output_dir, "simulation"),
            'save_trees': True,
            'scenario_weights': {
                'base': 0.4,
                'rapid_radiation': 0.3,
                'bottleneck': 0.2,
                'continuous_migration': 0.1
            }
        }
        
        # Save configuration
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Step 1: Run simulations
    if not args.skip_simulation:
        success = run_command(
            [sys.executable, simulation_script, "--config", config_path],
            "Simulations"
        )
        if not success:
            logging.error("Pipeline failed at simulation step")
            return
    else:
        logging.info("Skipping simulation step")
    
    # Results paths
    results_csv = f"{config['output_prefix']}_results.csv"
    
    # Step 2: Run statistical analysis
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    success = run_command(
        [sys.executable, analysis_script, "--data", results_csv, "--output", analysis_dir, "--ml"],
        "Statistical analysis"
    )
    if not success:
        logging.warning("Statistical analysis failed, continuing with pipeline")
    
    # Step 3: Create visualizations
    if not args.skip_visualization:
        visualization_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(visualization_dir, exist_ok=True)
        
        success = run_command(
            [sys.executable, visualization_script, "--data", results_csv, "--output", visualization_dir],
            "Visualizations"
        )
        if not success:
            logging.warning("Visualization failed, continuing with pipeline")
    else:
        logging.info("Skipping visualization step")
    
    # Step 4: Run validation tests
    if not args.skip_validation:
        validation_dir = os.path.join(output_dir, "validation")
        os.makedirs(validation_dir, exist_ok=True)
        
        success = run_command(
            [sys.executable, validation_script, "--pipeline", simulation_script, 
             "--output", validation_dir, "--test", "all"],
            "Validation tests"
        )
        if not success:
            logging.warning("Validation tests failed")
    else:
        logging.info("Skipping validation step")
    
    # Generate final report
    report_path = os.path.join(output_dir, "pipeline_report.md")
    with open(report_path, 'w') as f:
        f.write(f"# Introgression vs ILS Analysis Pipeline Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- Number of simulations: {config['num_simulations']}\n")
        f.write(f"- Genome length: {config['genome_length']}\n")
        f.write(f"- Scenario weights: {json.dumps(config.get('scenario_weights', {}))}\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("For detailed results, see the following directories:\n\n")
        f.write(f"- Simulation results: `{config['output_prefix']}_results.csv`\n")
        f.write(f"- Statistical analysis: `{analysis_dir}`\n")
        f.write(f"- Visualizations: `{visualization_dir}`\n")
        f.write(f"- Validation tests: `{validation_dir}`\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Review the statistical analysis to identify key patterns in the data\n")
        f.write("2. Examine the visualizations to understand parameter relationships\n")
        f.write("3. Evaluate the validation results to assess method robustness\n")
        f.write("4. Consider refining parameters based on findings\n")
    
    logging.info(f"Pipeline completed successfully! Results stored in {output_dir}")
    logging.info(f"See the pipeline report at {report_path}")

if __name__ == "__main__":
    main()