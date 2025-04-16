import torch
import numpy as np
import matplotlib.pyplot as plt
from functions import parse_args, gen_synthetic_data_fix
from exponential.algos import EMCov
from adaptive.algos import GaussCov, LapCov, SeparateCov, SeparateLapCov, AdaptiveCov, AdaptiveLapCov
from coinpress.algos import cov_est
import os
import matplotlib.ticker as ticker


class Seller:
    """
    Seller class that holds the seller's dataset and can produce
    differentially private covariance matrices using various DP algorithms.
    """
    def __init__(self, S, args):
        """
        Initialize the seller with a dataset.
        
        Args:
            S (torch.Tensor): The seller's dataset (shape: ns x d)
            args: Arguments containing privacy parameters
        """
        self.S = S
        self.ns, self.d = S.shape
        self.args = args
        # Compute true covariance
        self.CS = torch.mm(S.t(), S) / self.ns
        
    def get_private_cov(self, algorithm, rho=None, epsilon=None):
        """
        Generate a differentially private covariance matrix using the specified algorithm.
        
        Args:
            algorithm (str): Name of the DP algorithm to use
            rho (float, optional): Privacy parameter for zCDP
            epsilon (float, optional): Privacy parameter for pure DP
        
        Returns:
            torch.Tensor: Differentially private covariance matrix
        """
        if rho is not None:
            self.args.total_budget = rho
        
        # Clone the dataset to avoid modifying the original
        X = self.S.clone()
        
        # Use the specified algorithm
        if algorithm == 'EM':
            # Exponential Mechanism (approximate DP)
            cov_dp = EMCov(X, self.args, b_budget=False, b_fleig=True)
        elif algorithm == 'Gauss':
            # Gaussian Mechanism
            cov_dp = GaussCov(X, self.ns, self.d, self.args.total_budget, b_fleig=True)
        elif algorithm == 'Separate':
            # Separate noise for eigenvalues and eigenvectors
            cov_dp = SeparateCov(X, self.ns, self.d, self.args.total_budget, b_fleig=True)
        elif algorithm == 'Adaptive':
            # Adaptive privacy budget allocation
            cov_dp = AdaptiveCov(X, self.args)
        elif algorithm == 'CoinPress-1':
            # CoinPress with 1 iteration
            self.args.t = 1
            self.args.rho = [self.args.total_budget]
            cov_dp = cov_est(X, self.args)
        elif algorithm == 'CoinPress-2':
            # CoinPress with 2 iterations
            self.args.t = 2
            self.args.rho = [(1.0/4.0)*self.args.total_budget, (3.0/4.0)*self.args.total_budget]
            cov_dp = cov_est(X, self.args)
        elif algorithm == 'CoinPress-3':
            # CoinPress with 3 iterations
            self.args.t = 3
            self.args.rho = [(1.0/8.0)*self.args.total_budget, (1.0/8.0)*self.args.total_budget, (3.0/4.0)*self.args.total_budget]
            cov_dp = cov_est(X, self.args)
        elif algorithm == 'EM-Pure':
            # Exponential Mechanism (pure DP)
            cov_dp = EMCov(X, self.args, b_budget=True, b_fleig=True)
        elif algorithm == 'Lap':
            # Laplace Mechanism
            cov_dp = LapCov(X, self.ns, self.d, epsilon, b_fleig=True)
        elif algorithm == 'SeparateLap':
            # Separate Laplace noise
            cov_dp = SeparateLapCov(X, self.ns, self.d, epsilon, b_fleig=True)
        elif algorithm == 'AdaptiveLap':
            # Adaptive Laplace noise
            cov_dp = AdaptiveLapCov(X, self.args)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return cov_dp
    
    def get_true_cov(self):
        """Get the true (non-private) covariance matrix."""
        return self.CS


class Buyer:
    """
    Buyer class that can analyze private covariance matrices
    from the seller and compute diversity and relevance metrics.
    """
    def __init__(self, B):
        """
        Initialize the buyer with a dataset.
        
        Args:
            B (torch.Tensor): The buyer's dataset (shape: nb x d)
        """
        self.B = B
        self.nb, self.d = B.shape
        # Compute covariance matrix
        self.CB = torch.mm(B.t(), B) / self.nb
    
    def analyze_seller_cov(self, C_DP_S):
        """
        Analyze the seller's private covariance matrix.
        
        Args:
            C_DP_S (torch.Tensor): Seller's differentially private covariance matrix
        
        Returns:
            tuple: (eigenvalues, eigenvectors, projected eigenvalues)
        """
        # Get eigenvectors and eigenvalues of seller's private covariance
        eigenvalues, eigenvectors = torch.linalg.eigh(C_DP_S)
        
        # Sort in descending order (torch.linalg.eigh returns ascending order)
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)
        
        # Project buyer's covariance onto seller's eigenvectors
        projected_eigenvalues = torch.zeros(self.d)
        for i in range(self.d):
            u_i = eigenvectors[:, i]
            projected_eigenvalues[i] = torch.dot(u_i, torch.mv(self.CB, u_i))
        
        return eigenvalues, eigenvectors, projected_eigenvalues
    
    def compute_metrics(self, seller_eigenvalues, projected_eigenvalues):
        """
        Compute diversity and relevance metrics.
        
        Args:
            seller_eigenvalues (torch.Tensor): Eigenvalues from seller's cov matrix
            projected_eigenvalues (torch.Tensor): Buyer's projected eigenvalues
        
        Returns:
            tuple: (diversity, relevance)
        """
        # Ensure eigenvalues are positive for numerical stability
        seller_eigenvalues = torch.clamp(seller_eigenvalues, min=1e-10)
        projected_eigenvalues = torch.clamp(projected_eigenvalues, min=1e-10)
        
        # Diversity metric
        diversity_terms = torch.abs(projected_eigenvalues - seller_eigenvalues) / torch.max(
            projected_eigenvalues, seller_eigenvalues)
        diversity = torch.prod(diversity_terms) ** (1.0 / self.d)
        
        # Relevance metric
        relevance_terms = torch.min(projected_eigenvalues, seller_eigenvalues) / torch.max(
            projected_eigenvalues, seller_eigenvalues)
        relevance = torch.prod(relevance_terms) ** (1.0 / self.d)
        
        return diversity.item(), relevance.item()


def generate_datasets(ns, nb, d, s=3, N=4, correlation=0.5, seed=42):
    """
    Generate synthetic datasets for both seller and buyer.
    The datasets can optionally have some correlation.
    
    Args:
        ns (int): Number of samples for seller
        nb (int): Number of samples for buyer
        d (int): Dimensionality of the data
        s (float): Steepness in Zipf law
        N (int): Number of buckets in Zipf law
        correlation (float): Correlation between seller and buyer data (0.0-1.0)
        seed (int): Random seed
        
    Returns:
        tuple: (seller_data, buyer_data)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate seller's data
    S = gen_synthetic_data_fix(d, ns, s, N, seed=seed)
    
    # Generate buyer's data with some correlation to seller's data
    if correlation > 0:
        # Generate a base dataset
        B_base = gen_synthetic_data_fix(d, nb, s, N, seed=seed+1)
        
        # Mix seller's data with new data to create correlated buyer's data
        # Get the covariance of seller's data
        cov_S = torch.mm(S.t(), S) / ns
        
        # Create a distribution with this covariance
        mean = torch.zeros(d)
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov_S)
        
        # Sample from this distribution to create correlated data
        correlated_part = dist.sample((nb,))
        
        # Mix the two parts according to the correlation parameter
        B = correlation * correlated_part + (1 - correlation) * B_base
    else:
        # Generate independent data for buyer
        B = gen_synthetic_data_fix(d, nb, s, N, seed=seed+100)
    
    return S, B


def run_marketplace_experiment(ns=5000, nb=2000, d=20, rhos=[0.01, 0.1, 1.0, 10.0], 
                               correlation=0.5, seed=42, output_dir="results"):
    """
    Run a data marketplace experiment with all DP algorithms.
    
    Args:
        ns (int): Number of seller samples
        nb (int): Number of buyer samples
        d (int): Data dimensionality
        rhos (list): List of privacy budgets to test
        correlation (float): Correlation between buyer and seller data
        seed (int): Random seed
        output_dir (str): Directory to save the results
        
    Returns:
        dict: Results containing metrics for all algorithms and privacy budgets
    """
    # Generate datasets
    S, B = generate_datasets(ns, nb, d, correlation=correlation, seed=seed)
    
    # Setup args for privacy parameters
    args = parse_args()
    args.n = ns
    args.d = d
    args.delta = 1e-10  # For approximate DP
    args.beta = 0.1  # Probability bound
    args.u = 1.0  # Initial upper bound for covariance in CoinPress
    
    # Initialize the seller and buyer
    seller = Seller(S, args)
    buyer = Buyer(B)
    
    # Algorithms to test
    approx_dp_algorithms = ['EM', 'Gauss', 'Separate', 'Adaptive', 
                            'CoinPress-1', 'CoinPress-2', 'CoinPress-3']
    pure_dp_algorithms = ['EM-Pure', 'Lap', 'SeparateLap', 'AdaptiveLap']
    
    # Store results
    results = {
        'rhos': rhos,
        'epsilons': [np.sqrt(2*rho) for rho in rhos],
        'approx_dp': {alg: {'diversity': [], 'relevance': [], 'f_norm_error': []} for alg in approx_dp_algorithms},
        'pure_dp': {alg: {'diversity': [], 'relevance': [], 'f_norm_error': []} for alg in pure_dp_algorithms}
    }
    
    # True seller covariance
    true_cov = seller.get_true_cov()
    
    # Run experiments for each privacy budget
    for i, rho in enumerate(rhos):
        print(f"\nTesting with privacy budget rho = {rho}")
        epsilon = np.sqrt(2*rho)  # Convert to epsilon for pure DP
        
        # Test approximate DP algorithms
        for alg in approx_dp_algorithms:
            print(f"  Running {alg}...")
            # Get private covariance matrix
            cov_dp = seller.get_private_cov(alg, rho=rho)
            
            # Compute Frobenius norm error
            f_norm_error = torch.norm(true_cov - cov_dp, 'fro').item()
            results['approx_dp'][alg]['f_norm_error'].append(f_norm_error)
            
            # Buyer analyzes the private covariance
            seller_eigenvalues, _, projected_eigenvalues = buyer.analyze_seller_cov(cov_dp)
            
            # Compute metrics
            diversity, relevance = buyer.compute_metrics(seller_eigenvalues, projected_eigenvalues)
            results['approx_dp'][alg]['diversity'].append(diversity)
            results['approx_dp'][alg]['relevance'].append(relevance)
            
            print(f"    F-norm error: {f_norm_error:.6f}, Diversity: {diversity:.6f}, Relevance: {relevance:.6f}")
        
        # Test pure DP algorithms
        for alg in pure_dp_algorithms:
            print(f"  Running {alg}...")
            # Get private covariance matrix
            cov_dp = seller.get_private_cov(alg, epsilon=epsilon)
            
            # Compute Frobenius norm error
            f_norm_error = torch.norm(true_cov - cov_dp, 'fro').item()
            results['pure_dp'][alg]['f_norm_error'].append(f_norm_error)
            
            # Buyer analyzes the private covariance
            seller_eigenvalues, _, projected_eigenvalues = buyer.analyze_seller_cov(cov_dp)
            
            # Compute metrics
            diversity, relevance = buyer.compute_metrics(seller_eigenvalues, projected_eigenvalues)
            results['pure_dp'][alg]['diversity'].append(diversity)
            results['pure_dp'][alg]['relevance'].append(relevance)
            
            print(f"    F-norm error: {f_norm_error:.6f}, Diversity: {diversity:.6f}, Relevance: {relevance:.6f}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot results
    plot_results(results, output_dir)
    
    return results


def plot_results(results, output_dir):
    """
    Plot results of the experiment.
    
    Args:
        results (dict): Results from the experiment
        output_dir (str): Directory to save the plots
    """
    rhos = results['rhos']
    epsilons = results['epsilons']
    
    # Calculate total number of algorithms to ensure we have enough colors
    total_algorithms = len(results['approx_dp']) + len(results['pure_dp'])
    # Set up colors for different algorithms - ensure we have at least as many colors as algorithms
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, total_algorithms)))
    
    # Plot F-norm error for approximate DP algorithms
    plt.figure(figsize=(12, 8))
    for i, (alg, data) in enumerate(results['approx_dp'].items()):
        plt.plot(rhos, data['f_norm_error'], marker='o', color=colors[i], label=alg)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Privacy Budget (ρ)')
    plt.ylabel('Frobenius Norm Error')
    plt.title('Error vs. Privacy Budget (Approximate DP)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/approx_dp_error.png")
    
    # Plot F-norm error for pure DP algorithms
    plt.figure(figsize=(12, 8))
    for i, (alg, data) in enumerate(results['pure_dp'].items()):
        plt.plot(epsilons, data['f_norm_error'], marker='o', color=colors[i], label=alg)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('Frobenius Norm Error')
    plt.title('Error vs. Privacy Budget (Pure DP)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pure_dp_error.png")
    
    # Plot diversity for approximate DP algorithms
    plt.figure(figsize=(12, 8))
    for i, (alg, data) in enumerate(results['approx_dp'].items()):
        plt.plot(rhos, data['diversity'], marker='o', color=colors[i], label=alg)
    
    plt.xscale('log')
    plt.xlabel('Privacy Budget (ρ)')
    plt.ylabel('Diversity')
    plt.title('Diversity vs. Privacy Budget (Approximate DP)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/approx_dp_diversity.png")
    
    # Plot diversity for pure DP algorithms
    plt.figure(figsize=(12, 8))
    for i, (alg, data) in enumerate(results['pure_dp'].items()):
        plt.plot(epsilons, data['diversity'], marker='o', color=colors[i], label=alg)
    
    plt.xscale('log')
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('Diversity')
    plt.title('Diversity vs. Privacy Budget (Pure DP)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pure_dp_diversity.png")
    
    # Plot relevance for approximate DP algorithms
    plt.figure(figsize=(12, 8))
    for i, (alg, data) in enumerate(results['approx_dp'].items()):
        plt.plot(rhos, data['relevance'], marker='o', color=colors[i], label=alg)
    
    plt.xscale('log')
    plt.xlabel('Privacy Budget (ρ)')
    plt.ylabel('Relevance')
    plt.title('Relevance vs. Privacy Budget (Approximate DP)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/approx_dp_relevance.png")
    
    # Plot relevance for pure DP algorithms
    plt.figure(figsize=(12, 8))
    for i, (alg, data) in enumerate(results['pure_dp'].items()):
        plt.plot(epsilons, data['relevance'], marker='o', color=colors[i], label=alg)
    
    plt.xscale('log')
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('Relevance')
    plt.title('Relevance vs. Privacy Budget (Pure DP)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pure_dp_relevance.png")
    
    # Plot privacy-utility tradeoff (error vs. diversity)
    plt.figure(figsize=(12, 8))
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    # Approximate DP algorithms
    for i, (alg, data) in enumerate(results['approx_dp'].items()):
        plt.scatter(data['f_norm_error'], data['diversity'], 
                   s=100, marker=markers[i % len(markers)], color=colors[i], 
                   label=f"{alg} (ρ)", alpha=0.7)
        # Add annotations for privacy budget
        for j, rho in enumerate(rhos):
            plt.annotate(f"ρ={rho}", 
                        (data['f_norm_error'][j], data['diversity'][j]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
    
    # Pure DP algorithms
    for i, (alg, data) in enumerate(results['pure_dp'].items()):
        # Use modulo to ensure we don't go out of bounds with the color index
        color_idx = (i+len(results['approx_dp'])) % len(colors)
        plt.scatter(data['f_norm_error'], data['diversity'], 
                   s=100, marker=markers[i % len(markers)], color=colors[color_idx], 
                   label=f"{alg} (ε)", alpha=0.7)
        # Add annotations for privacy budget
        for j, eps in enumerate(epsilons):
            plt.annotate(f"ε={eps:.2f}", 
                        (data['f_norm_error'][j], data['diversity'][j]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
    
    plt.xscale('log')
    plt.xlabel('Frobenius Norm Error')
    plt.ylabel('Diversity')
    plt.title('Privacy-Utility Tradeoff: Error vs. Diversity')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/privacy_utility_tradeoff.png")


if __name__ == "__main__":
    # Define experiment parameters
    ns = 5000  # Number of seller samples
    nb = 2000  # Number of buyer samples
    d = 20     # Data dimensionality
    rhos = [0.01, 0.1, 1.0, 10.0]  # Privacy budgets to test
    correlation = 0.5  # Correlation between buyer and seller data
    seed = 42  # Random seed
    output_dir = "marketplace_results"  # Output directory
    
    # Run the experiment
    print(f"Running data marketplace experiment with ns={ns}, nb={nb}, d={d}")
    results = run_marketplace_experiment(
        ns=ns, nb=nb, d=d, rhos=rhos, 
        correlation=correlation, seed=seed, 
        output_dir=output_dir
    )
    
    print(f"\nExperiment complete. Results saved to {output_dir}/")