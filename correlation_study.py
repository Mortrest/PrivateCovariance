import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from dp_marketplace import generate_datasets, Seller, Buyer, parse_args

def correlation_study(ns=5000, nb=2000, d=20, rho=1.0, correlations=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                      seed=42, output_dir="correlation_results"):
    """
    Study the effect of data correlation between buyer and seller on the diversity and relevance metrics.
    
    Args:
        ns (int): Number of seller samples
        nb (int): Number of buyer samples
        d (int): Data dimensionality
        rho (float): Privacy budget
        correlations (list): List of correlation values to test
        seed (int): Random seed
        output_dir (str): Directory to save the results
    
    Returns:
        dict: Results containing metrics for all algorithms and correlation values
    """
    # Setup args for privacy parameters
    args = parse_args()
    args.n = ns
    args.d = d
    args.delta = 1e-10  # For approximate DP
    args.beta = 0.1  # Probability bound
    args.total_budget = rho
    epsilon = np.sqrt(2*rho)  # For pure DP
    
    # Algorithms to test
    approx_dp_algorithms = ['EM', 'Gauss', 'Separate', 'Adaptive', 'CoinPress-2']
    pure_dp_algorithms = ['EM-Pure', 'Lap', 'SeparateLap', 'AdaptiveLap']
    
    # Store results
    results = {
        'correlations': correlations,
        'approx_dp': {alg: {'diversity': [], 'relevance': []} for alg in approx_dp_algorithms},
        'pure_dp': {alg: {'diversity': [], 'relevance': []} for alg in pure_dp_algorithms}
    }
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run experiments for each correlation value
    for corr in correlations:
        print(f"\nTesting with correlation = {corr}")
        
        # Generate datasets with the specified correlation
        S, B = generate_datasets(ns, nb, d, correlation=corr, seed=seed)
        
        # Initialize seller and buyer
        seller = Seller(S, args)
        buyer = Buyer(B)
        
        # Get true covariance for reference
        true_cov = seller.get_true_cov()
        
        # Test approximate DP algorithms
        for alg in approx_dp_algorithms:
            print(f"  Running {alg}...")
            
            # Get private covariance matrix
            cov_dp = seller.get_private_cov(alg, rho=rho)
            
            # Buyer analyzes the private covariance
            seller_eigenvalues, _, projected_eigenvalues = buyer.analyze_seller_cov(cov_dp)
            
            # Compute metrics
            diversity, relevance = buyer.compute_metrics(seller_eigenvalues, projected_eigenvalues)
            results['approx_dp'][alg]['diversity'].append(diversity)
            results['approx_dp'][alg]['relevance'].append(relevance)
            
            print(f"    Diversity: {diversity:.6f}, Relevance: {relevance:.6f}")
        
        # Test pure DP algorithms
        for alg in pure_dp_algorithms:
            print(f"  Running {alg}...")
            
            # Get private covariance matrix
            cov_dp = seller.get_private_cov(alg, epsilon=epsilon)
            
            # Buyer analyzes the private covariance
            seller_eigenvalues, _, projected_eigenvalues = buyer.analyze_seller_cov(cov_dp)
            
            # Compute metrics
            diversity, relevance = buyer.compute_metrics(seller_eigenvalues, projected_eigenvalues)
            results['pure_dp'][alg]['diversity'].append(diversity)
            results['pure_dp'][alg]['relevance'].append(relevance)
            
            print(f"    Diversity: {diversity:.6f}, Relevance: {relevance:.6f}")
    
    # Plot results
    plot_correlation_results(results, output_dir, rho, epsilon)
    
    return results


def plot_correlation_results(results, output_dir, rho, epsilon):
    """
    Plot the results of the correlation study.
    
    Args:
        results (dict): Results from the experiment
        output_dir (str): Directory to save the plots
        rho (float): Privacy budget used (rho)
        epsilon (float): Privacy budget used (epsilon)
    """
    correlations = results['correlations']
    
    # Set up colors for different algorithms
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot diversity vs. correlation for approximate DP algorithms
    plt.figure(figsize=(12, 8))
    for i, (alg, data) in enumerate(results['approx_dp'].items()):
        plt.plot(correlations, data['diversity'], marker='o', color=colors[i], label=alg)
    
    plt.xlabel('Correlation between Buyer and Seller Data')
    plt.ylabel('Diversity')
    plt.title(f'Diversity vs. Correlation (Approximate DP, ρ={rho})')
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/approx_dp_diversity_vs_correlation.png")
    
    # Plot diversity vs. correlation for pure DP algorithms
    plt.figure(figsize=(12, 8))
    for i, (alg, data) in enumerate(results['pure_dp'].items()):
        plt.plot(correlations, data['diversity'], marker='o', color=colors[i], label=alg)
    
    plt.xlabel('Correlation between Buyer and Seller Data')
    plt.ylabel('Diversity')
    plt.title(f'Diversity vs. Correlation (Pure DP, ε={epsilon:.2f})')
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pure_dp_diversity_vs_correlation.png")
    
    # Plot relevance vs. correlation for approximate DP algorithms
    plt.figure(figsize=(12, 8))
    for i, (alg, data) in enumerate(results['approx_dp'].items()):
        plt.plot(correlations, data['relevance'], marker='o', color=colors[i], label=alg)
    
    plt.xlabel('Correlation between Buyer and Seller Data')
    plt.ylabel('Relevance')
    plt.title(f'Relevance vs. Correlation (Approximate DP, ρ={rho})')
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/approx_dp_relevance_vs_correlation.png")
    
    # Plot relevance vs. correlation for pure DP algorithms
    plt.figure(figsize=(12, 8))
    for i, (alg, data) in enumerate(results['pure_dp'].items()):
        plt.plot(correlations, data['relevance'], marker='o', color=colors[i], label=alg)
    
    plt.xlabel('Correlation between Buyer and Seller Data')
    plt.ylabel('Relevance')
    plt.title(f'Relevance vs. Correlation (Pure DP, ε={epsilon:.2f})')
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pure_dp_relevance_vs_correlation.png")


if __name__ == "__main__":
    # Parameters
    ns = 5000  # Number of seller samples
    nb = 2000  # Number of buyer samples
    d = 20     # Data dimensionality
    rho = 1.0  # Privacy budget
    correlations = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Correlation values to test
    seed = 42  # Random seed
    output_dir = "correlation_results"  # Output directory
    
    print(f"Running correlation study with ns={ns}, nb={nb}, d={d}, rho={rho}")
    results = correlation_study(
        ns=ns, nb=nb, d=d, rho=rho, 
        correlations=correlations, seed=seed, 
        output_dir=output_dir
    )
    
    print(f"\nStudy complete. Results saved to {output_dir}/")