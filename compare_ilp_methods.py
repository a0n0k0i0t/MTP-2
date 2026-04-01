import os
import sys
import time
import subprocess
import argparse
import re

def run_python_ilp(script, data_file):
    if not os.path.exists(script):
        print(f"Error: Script {script} not found.")
        return 0.0, 0.0, "Script not found"

    print(f"Running {script}...")
    start = time.time()
    try:
        result = subprocess.run(["python", script, data_file], capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"  -> Timeout! Exceeded 300 seconds.")
        return "5min+", 0.0, ""
        
    end = time.time()
    
    out = result.stdout
    sim_match = re.search(r"Jaccard Similarity:\s*([\d.]+)", out)
    sim = float(sim_match.group(1)) if sim_match else 0.0
    
    # Python scripts might also output "Time taken"
    time_match = re.search(r"Time taken.+?([\d.]+)\s*seconds", out, re.IGNORECASE)
    exec_time = float(time_match.group(1)) if time_match else (end - start)

    return exec_time, sim, out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare execution time and similarity of multiple ILP methods.")
    parser.add_argument("-f", "--file", type=str, default="points.txt", help="Path to the dataset file")
    args = parser.parse_args()

    data_file = args.file
    
    if not os.path.exists(data_file):
        print(f"Error: Could not find dataset '{data_file}'")
        sys.exit(1)

    print(f"=== Evaluating ILP Methods on Dataset: {data_file} ===\n")

    methods = [
        ("ILP Dinkelbach Warm Start", "solve_ilp_Dinkleback_gamma_warm_start.py"),
        ("Dynamic Grid Warm Start", "solve_ilp_dynamic_grid_warm_start.py"),
        ("Dynamic Grid MIPFocus", "solve_ilp_dynamic_grid_warm_start_mipfocus.py"),
        ("Dynamic Grid Dinkelbach Warm", "solve_ilp_dynamic_grid_dinkelbach_warm_start.py")
    ]

    results = []

    for name, script in methods:
        exec_time, sim, out = run_python_ilp(script, data_file)
        results.append((name, exec_time, sim))

    print("\n" + "="*80)
    print(f"{'Algorithm':<40} | {'Execution Time (s)':<20} | {'Jaccard Similarity':<20}")
    print("-" * 80)
    for name, exec_time, sim in results:
        exec_str = f"{exec_time:<20.4f}" if isinstance(exec_time, float) else f"{exec_time:<20}"
        print(f"{name:<40} | {exec_str} | {sim:<20.4f}")
    print("="*80)
