import torch
import numpy as np
import matplotlib.pyplot as plt
from functions import parse_args, gen_synthetic_data_fix
from exponential.algos import EMCov
from adaptive.algos import GaussCov, LapCov, SeparateCov, SeparateLapCov, AdaptiveCov, AdaptiveLapCov
from coinpress.algos import cov_est

def test_all_algorithms():
    # Initialize parameters
    args = parse_args()
    args.d = 20  # dimension
    args.n = 5000  # number of samples
    args.N = 4  # number of buckets in Zipf law (from default in parse_args)
    args.s = 3  # steepness in Zipf law (from default in parse_args)
    args.delta = 1e-10  # for approximate DP
    args.beta = 0.1  # probability bound
    
    # Set privacy budgets for testing
    rhos = [0.01, 0.1, 1.0, 10.0]
    epsilons = [np.sqrt(2*rho) for rho in rhos]  # Convert rho to epsilon for pure DP
    
    # Create synthetic data
    seed = 42  # For reproducibility
    X = gen_synthetic_data_fix(args.d, args.n, args.s, args.N, seed=seed)
    
    # Compute true covariance (non-private)
    true_cov = torch.mm(X.t(), X) / args.n
    print(f"Trace of true covariance: {float(torch.trace(true_cov))}")
    
    # Store results: privacy budget -> algorithm -> error
    results = {rho: {} for rho in rhos}
    
    # Test all algorithms for each privacy budget
    for i, rho in enumerate(rhos):
        args.total_budget = rho
        print(f"\nTesting with privacy budget rho = {rho} (epsilon = {epsilons[i]:.4f})")
        
        # Configure CoinPress iterations
        Ps1 = [args.total_budget]
        Ps2 = [(1.0/4.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        Ps3 = [(1.0/8.0)*args.total_budget, (1.0/8.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        Ps4 = [(1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        Ps5 = [(1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (3.0/4.0)*args.total_budget]
        
        # Run algorithms (approximate DP)
        print("Running approximate DP algorithms...")
        cov_em = EMCov(X.clone(), args, b_budget=False, b_fleig=True)
        cov_gauss = GaussCov(X.clone(), args.n, args.d, rho, b_fleig=True)
        cov_sep = SeparateCov(X.clone(), args.n, args.d, rho, b_fleig=True)
        cov_adapt = AdaptiveCov(X.clone(), args)
        
        args.t = 1
        args.rho = Ps1
        cov_cpt1 = cov_est(X.clone(), args)
        args.t = 2
        args.rho = Ps2
        cov_cpt2 = cov_est(X.clone(), args)
        args.t = 3
        args.rho = Ps3
        cov_cpt3 = cov_est(X.clone(), args)
        args.t = 4
        args.rho = Ps4
        cov_cpt4 = cov_est(X.clone(), args)
        args.t = 5
        args.rho = Ps5
        cov_cpt5 = cov_est(X.clone(), args)
        
        # Run algorithms (pure DP with epsilon)
        print("Running pure DP algorithms...")
        eps = epsilons[i]
        cov_em_pure = EMCov(X.clone(), args, b_budget=True, b_fleig=True)
        cov_lap = LapCov(X.clone(), args.n, args.d, eps, b_fleig=True)
        cov_sep_lap = SeparateLapCov(X.clone(), args.n, args.d, eps, b_fleig=True)
        cov_adapt_lap = AdaptiveLapCov(X.clone(), args)
        
        # Compute errors (Frobenius norm)
        results[rho] = {
            # Approximate DP
            'Exponential': float(torch.norm(true_cov - cov_em, 'fro')),
            'Gaussian': float(torch.norm(true_cov - cov_gauss, 'fro')),
            'Separate': float(torch.norm(true_cov - cov_sep, 'fro')),
            'Adaptive': float(torch.norm(true_cov - cov_adapt, 'fro')),
            'CoinPress-1': float(torch.norm(true_cov - cov_cpt1, 'fro')),
            'CoinPress-2': float(torch.norm(true_cov - cov_cpt2, 'fro')),
            'CoinPress-3': float(torch.norm(true_cov - cov_cpt3, 'fro')),
            'CoinPress-4': float(torch.norm(true_cov - cov_cpt4, 'fro')),
            'CoinPress-5': float(torch.norm(true_cov - cov_cpt5, 'fro')),
            # Pure DP
            'Exponential-Pure': float(torch.norm(true_cov - cov_em_pure, 'fro')),
            'Laplace': float(torch.norm(true_cov - cov_lap, 'fro')),
            'SeparateLap': float(torch.norm(true_cov - cov_sep_lap, 'fro')),
            'AdaptiveLap': float(torch.norm(true_cov - cov_adapt_lap, 'fro')),
            # Baseline (zero)
            'Zero': float(torch.norm(true_cov, 'fro'))
        }
        
        # Print errors
        for alg, err in results[rho].items():
            print(f"{alg}: {err:.6f}")
    
    return results, rhos

def plot_results(results, rhos):
    # Plot results for approximate DP algorithms
    plt.figure(figsize=(12, 8))
    algorithms = ['Exponential', 'Gaussian', 'Separate', 'Adaptive', 
                  'CoinPress-1', 'CoinPress-2', 'CoinPress-3', 'CoinPress-4', 'CoinPress-5']
    
    for alg in algorithms:
        errors = [results[rho][alg] for rho in rhos]
        plt.plot(rhos, errors, marker='o', label=alg)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Privacy Budget (ρ)')
    plt.ylabel('Frobenius Norm Error')
    plt.title('Performance of Approximate DP Algorithms (d=20, n=5000)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('approx_dp_results.png')
    
    # Plot results for pure DP algorithms
    plt.figure(figsize=(12, 8))
    algorithms = ['Exponential-Pure', 'Laplace', 'SeparateLap', 'AdaptiveLap']
    
    for alg in algorithms:
        errors = [results[rho][alg] for rho in rhos]
        plt.plot(rhos, errors, marker='o', label=alg)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Privacy Budget (ρ)')
    plt.ylabel('Frobenius Norm Error')
    plt.title('Performance of Pure DP Algorithms (d=20, n=5000)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('pure_dp_results.png')

if __name__ == "__main__":
    print("Testing all DP covariance estimation algorithms on synthetic data")
    print("Dimension: 20, Sample size: 5000")
    results, rhos = test_all_algorithms()
    plot_results(results, rhos)
    print("\nResults saved to approx_dp_results.png and pure_dp_results.png")