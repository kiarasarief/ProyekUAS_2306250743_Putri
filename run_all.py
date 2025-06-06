# run_all.py - Script to run all components of the ECG signal decomposition analysis
import os
import subprocess
import time
import sys

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def run_command(cmd, description=None):
    """Run a command and display its output"""
    if description:
        print(f"{description}...")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.returncode != 0:
        print(f"Error: Command failed with code {result.returncode}")
        if result.stderr:
            print(f"Error details: {result.stderr}")
        return False
    
    return True

def main():
    print_header("ECG Signal Decomposition Analysis")
    
    start_time = time.time()
    
    # Step 1: Compile the C++ code
    print_header("Step 1: Compiling C++ code")
    success = run_command("g++ main.cpp -o main.exe -std=c++11", "Compiling C++ code")
    if not success:
        sys.exit(1)
    # Step 2: Run the C++ program
    print_header("Step 2: Running ECG signal decomposition")
    success = run_command("main.exe", "Processing ECG signals")
    if not success:
        sys.exit(1)
    
    # Step 3: Create plots directory
    if not os.path.exists("plots"):
        os.makedirs("plots")
        print("Created 'plots' directory for output visualizations")
    
    # Step 4: Run the basic visualization
    print_header("Step 3: Generating basic visualizations")
    success = run_command("python plot.py", "Running basic visualization")
    if not success:
        sys.exit(1)
    
    # Step 5: Run the advanced analysis
    print_header("Step 4: Running advanced analysis")
    success = run_command("python advanced_analysis.py", "Analyzing signal components and residuals")
    if not success:
        print("Warning: Advanced analysis failed, continuing...")
      # Step 6: Run the component analysis
    print_header("Step 5: Analyzing component contributions")
    success = run_command("python component_analysis.py", "Analyzing coefficient contributions and SNR")
    if not success:
        print("Warning: Component analysis failed, continuing...")
        
    # Step 7: Run the noise analysis
    print_header("Step 6: Analyzing noise performance")
    success = run_command("python noise_analysis.py", "Analyzing L1 vs L2 performance under different noise conditions")
    if not success:
        print("Warning: Noise analysis failed, continuing...")
    
    # Final report
    elapsed_time = time.time() - start_time
    print_header("Analysis Complete")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print("\nResults:")
    print("1. CSV data files in the working directory")
    print("2. Visualization images in the 'plots' directory")
    print("\nTo view the plots, check the 'plots' folder or run each script individually:")
    print("  - plot.py: Basic signal visualization")
    print("  - advanced_analysis.py: Residual and error analysis")
    print("  - component_analysis.py: Component contribution and SNR analysis")
    print("  - noise_analysis.py: L1 vs L2 performance under different noise levels")

if __name__ == "__main__":
    main()
