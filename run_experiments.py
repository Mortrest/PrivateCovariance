import torch
import numpy as np
import argparse
from dp_marketplace import run_marketplace_experiment

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run DP data marketplace experiments")
    parser.add_argument("--ns", type=int, default=5000, help="Number of seller samples")
    parser.add_argument("--nb", type=int, default=2000, help="Number of buyer samples")
    parser.add_argument("--d", type=int, default=20, help="Data dimensionality")
    parser.add_argument("--rho_min", type=float, default=0.01, help="Minimum privacy budget (rho)")
    parser.add_argument("--rho_max", type=float, default=10.0, help="Maximum privacy budget (rho)")
    parser.add_argument("--num_rhos", type=int, default=4, help="Number of privacy budgets to test")
    parser.add_argument("--correlation", type=float, default=0.5, help="Correlation between buyer and seller data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="marketplace_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Generate logarithmically spaced privacy budgets
    rhos = np.logspace(np.log10(args.rho_min), np.log10(args.rho_max), args.num_rhos)
    rhos = [float(rho) for rho in rhos]  # Convert to regular Python floats
    
    print(f"Running experiment with the following parameters:")
    print(f"  Seller samples (ns): {args.ns}")
    print(f"  Buyer samples (nb): {args.nb}")
    print(f"  Dimensionality (d): {args.d}")
    print(f"  Privacy budgets (rho): {rhos}")
    print(f"  Correlation: {args.correlation}")
    print(f"  Seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")
    
    # Run the experiment
    results = run_marketplace_experiment(
        ns=args.ns,
        nb=args.nb,
        d=args.d,
        rhos=rhos,
        correlation=args.correlation,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print("\nExperiment complete!")
    print(f"Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()