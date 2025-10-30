import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 

def chirp_mass_m1(m1, q):
    '''
    Calculate the chirp mass given primary mass m1 and mass ratio q
    '''
    return ((m1 ** 2 * q) ** (3 / 5)) / ((m1 * (1 + q)) ** (1 / 5))

def chirp_mass_m2(m2, q):
    '''
    Calculate the chirp mass given secondary mass m2 and mass ratio q
    '''
    return ((m2 ** 2 / q) ** (3 / 5)) / ((m2 * (1 + 1 / q)) ** (1 / 5))


def propose(m_prev, sigma_m, m_low, m_high):
    """
    Propose a new parameter 'm' with Gaussian random walk and reflection at boundaries
    """
    m_prop = np.random.normal(m_prev, sigma_m)

    # Reflect proposals outside of bounds
    if m_prop > m_high:
        m_prop = 2 * m_high - m_prop
    if m_prop < m_low:
        m_prop = 2 * m_low - m_prop

    return m_prop


def log_likelihood(M, M_cnn, M_error):
    """
    Evaluate the log likelihood assuming Gaussian errors
    L(m|M_cnn) ∝ exp(-(M - M_cnn)²/(2σ²))
    """
    return -0.5 * np.log(2.0 * np.pi * M_error**2) - ((M - M_cnn) ** 2) / (2 * M_error**2)
    

def log_posterior(m, q, M_cnn, M_error, mass_type):
    """
    Evaluate the log posterior: log P(m|data) ∝ log L(data|m)
    Prior is uniform and constant → ignored in Metropolis ratio
    """
    if mass_type == 'm1':
        M = chirp_mass_m1(m, q)
    else:
        M = chirp_mass_m2(m, q)
    
    return log_likelihood(M, M_cnn, M_error)


def MCMC(M_cnn, M_error, q, m_type, m_low=10.0, m_high=90.0, N_iter=50000, N_burnin=3000, seed=None):
    """
    Run MCMC to estimate m1 or m2 given CNN predictions of chirp mass and mass ratio
    
    Parameters:
    -----------
    M_cnn : float
        Chirp mass predicted by CNN (M☉)
    M_error : float
        Uncertainty on chirp mass (M☉)
    q : float
        Mass ratio predicted by CNN (q = m2/m1, q ≤ 1)
    m_type : str
        'm1' or 'm2' - which mass to estimate
    m_low, m_high : float
        Prior bounds on mass (M☉)
    N_iter : int
        Number of MCMC iterations (after burn-in)
    N_burnin : int
        Number of burn-in iterations to discard
    seed : int or None
        Random seed for reproducibility
    
    Returns:
    --------
    m_chain : ndarray
        MCMC chain of mass samples (shape: (N_iter, 1))
    """
    
    if seed is not None:
        np.random.seed(seed)

    # Proposal step size
    sigma_m = 0.02 * (m_high - m_low)

    # Initialize chain
    N_total = N_iter + N_burnin
    m_chain = np.zeros((N_total, 1))

    # Random initialization
    m_prev = np.random.uniform(m_low, m_high)

    # MCMC loop
    n_accepted = 0
    for i in tqdm(range(N_total)):
        # Propose new value
        m_prop = propose(m_prev, sigma_m, m_low, m_high)

        # Calculate log posteriors
        P_prop = log_posterior(m_prop, q, M_cnn, M_error, m_type)
        P_prev = log_posterior(m_prev, q, M_cnn, M_error, m_type)

        # Metropolis acceptance criterion
        log_ratio = P_prop - P_prev
        if np.log(np.random.uniform(0.0, 1.0)) < log_ratio:
            m_chain[i, :] = m_prop
            m_prev = m_prop
            n_accepted += 1
        else:
            m_chain[i, :] = m_prev

    # Remove burn-in
    m_chain = m_chain[N_burnin:, :]
    
    # Print diagnostics
    acceptance_rate = n_accepted / N_total
    print(f"MCMC for {m_type}:")
    print(f"  Mean: {np.mean(m_chain):.2f} M☉")
    print(f"  Median: {np.median(m_chain):.2f} M☉")
    print(f"  Std: {np.std(m_chain):.2f} M☉")

    return m_chain


def estimate_masses(M_cnn, q_cnn, M_error, plot=True):
    """
    Estimate m1 and m2 with uncertainties using MCMC
    
    Parameters:
    -----------
    M_cnn : float
        Chirp mass from CNN (M☉)
    q_cnn : float  
        Mass ratio from CNN (m2/m1, ≤ 1)
    M_error : float
        Uncertainty on chirp mass (M☉)
    plot : bool
        Whether to plot posterior distributions
    
    Returns:
    --------
    results : dict
        Dictionary with mass estimates and credible intervals
    """
    
    # Run MCMC for both masses
    m1_chain = MCMC(M_cnn, M_error, q_cnn, 'm1', seed=42)
    m2_chain = MCMC(M_cnn, M_error, q_cnn, 'm2', seed=43)
    
    # Calculate statistics
    m1_mean = np.mean(m1_chain)
    m1_median = np.median(m1_chain)
    m1_CI_90 = np.percentile(m1_chain, [5, 95])
    
    m2_mean = np.mean(m2_chain)
    m2_median = np.median(m2_chain)
    m2_CI_90 = np.percentile(m2_chain, [5, 95])
    
    results = {
        'm1_mean': m1_mean,
        'm1_median': m1_median,
        'm1_CI_90': m1_CI_90,
        'm2_mean': m2_mean,
        'm2_median': m2_median,
        'm2_CI_90': m2_CI_90,
    }
    
    # Plot posteriors
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # m1 posterior
        ax1.hist(m1_chain, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax1.axvline(m1_median, color='red', linewidth=2, label=f'Median: {m1_median:.2f} M☉')
        ax1.axvline(m1_CI_90[0], color='red', linestyle='--', linewidth=1, label=f'90% CI: [{m1_CI_90[0]:.2f}, {m1_CI_90[1]:.2f}]')
        ax1.axvline(m1_CI_90[1], color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('m₁ (M☉)')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Primary Mass Posterior')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # m2 posterior
        ax2.hist(m2_chain, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax2.axvline(m2_median, color='blue', linewidth=2, label=f'Median: {m2_median:.2f} M☉')
        ax2.axvline(m2_CI_90[0], color='blue', linestyle='--', linewidth=1, label=f'90% CI: [{m2_CI_90[0]:.2f}, {m2_CI_90[1]:.2f}]')
        ax2.axvline(m2_CI_90[1], color='blue', linestyle='--', linewidth=1)
        ax2.set_xlabel('m₂ (M☉)')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Secondary Mass Posterior')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return results