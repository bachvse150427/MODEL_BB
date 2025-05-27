import os
import sys
import subprocess
import time
import argparse
from datetime import datetime

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_script(script_name, description, *args):
    log_message(f"Starting {description}...")
    start_time = time.time()
    
    try:
        cmd = [sys.executable, script_name] + list(args)
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        log_message(f"Completed {description} in {elapsed_time:.2f} seconds")
        log_message(f"Output: {result.stdout[:500]}..." if len(result.stdout) > 500 else f"Output: {result.stdout}")
        
        return True
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        log_message(f"Error in {description} after {elapsed_time:.2f} seconds")
        log_message(f"Error: {e}")
        log_message(f"Stderr: {e.stderr}")
        
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Stock market prediction pipeline")
    parser.add_argument("--skip-optimizer", action="store_true", help="Skip the model optimization step")
    parser.add_argument("--skip-training", action="store_true", help="Skip the model training step")
    parser.add_argument("--skip-predictor", action="store_true", help="Skip the model prediction step")
    return parser.parse_args()

def main():
    args = parse_arguments()
    log_message("Starting complete model pipeline")
    
    # Step 1: Run optimizer.py (unless skipped)
    if args.skip_optimizer:
        log_message("Skipping model optimization as requested")
    else:
        if not run_script("optimizer.py", "model optimization"):
            log_message("Model optimization failed, stopping pipeline")
            return False
    
    # Step 2: Run trainer.py
    if args.skip_training:
        log_message("Skipping model training as requested")
    else:
        if not run_script("trainer.py", "model training"):
            log_message("Model training failed, stopping pipeline")
            return False
    
    # Step 3: Run predictor.py
    if args.skip_predictor:
        log_message("Skipping model prediction as requested")
    else:
        if not run_script("predictor.py", "model prediction"):
            log_message("Model prediction failed, stopping pipeline")
            return False
    
    # Step 4: Run inferencer.py with --all-tickers-all-data flag
    log_message("Starting inference for all data points of all tickers")
    if not run_script("inferencer.py", "model inference", "--all"):
        log_message("Model inference failed, stopping pipeline")
        return False
    
    # Step 5: Run push_mongo_data.py
    if not run_script("push_mongo_data.py", "database upload"):
        log_message("Database upload failed, stopping pipeline")
        return False
    
    log_message("Complete model pipeline executed successfully")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
