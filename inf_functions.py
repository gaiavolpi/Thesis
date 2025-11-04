import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
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


def plot_posteriors(m1_chain, m2_chain, m1_true, m2_true, synthetic=True):

    # Calculate statistics
    m1_median = np.median(m1_chain)
    m1_CI_90 = np.percentile(m1_chain, [5, 95])
    
    m2_median = np.median(m2_chain)
    m2_CI_90 = np.percentile(m2_chain, [5, 95])

    m1 = m1_chain.flatten()
    m2 = m2_chain.flatten()

    kde = gaussian_kde(np.vstack([m1, m2]))

    xmin, xmax = m1.min(), m1.max()
    ymin, ymax = m2.min(), m2.max()

    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = kde(positions).reshape(xx.shape)


    fig = plt.figure(figsize=(10,10))
    grid = plt.GridSpec(4,4, hspace=0.05, wspace=0.05)

    ax_main  = fig.add_subplot(grid[1:4, 0:3])
    ax_top   = fig.add_subplot(grid[0,     0:3], sharex=ax_main)
    ax_right = fig.add_subplot(grid[1:4,     3], sharey=ax_main)

    ax_main.contourf(xx, yy, f, levels=15, alpha=0.6, cmap="Blues")
    ax_main.contour(xx, yy, f, colors='navy', linewidths=0.8)

    ax_main.scatter(m1_true, m2_true, color="red", s=100, marker="*", label="True values" if synthetic else "LVK estimates")
    ax_main.scatter(m1_median, m2_median, color="yellowgreen", s=100, marker="*", label="Posterior median")

    # Lines
    ax_main.axvline(m1_true, color="red", linestyle="-", lw=0.7)
    ax_main.axhline(m2_true, color="red", linestyle="-", lw=0.7)

    ax_main.axvline(m1_median, color="yellowgreen", linestyle="-", lw=0.7)
    ax_main.axhline(m2_median, color="yellowgreen", linestyle="-", lw=0.7)

    ax_main.axvline(m1_CI_90[0], color="gray", linestyle="--", lw=0.7)
    ax_main.axvline(m1_CI_90[1], color="gray", linestyle="--", lw=0.7)
    ax_main.axhline(m2_CI_90[0], color="gray", linestyle="--", lw=0.7)
    ax_main.axhline(m2_CI_90[1], color="gray", linestyle="--", lw=0.7)


    # --- Marginal: m1 ---
    ax_top.hist(m1, bins=40, density=True, alpha=0.7, color="steelblue")
    ax_top.axvline(m1_true, color="red", linestyle="-", lw=0.7)
    ax_top.axvline(m1_median, color="yellowgreen", linestyle="-", lw=0.7)
    ax_top.axvline(m1_CI_90[0], color="gray", linestyle="--", lw=0.7)
    ax_top.axvline(m1_CI_90[1], color="gray", linestyle="--", lw=0.7)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # --- Marginal: m2 ---
    ax_right.hist(m2, bins=40, density=True, alpha=0.7, orientation="horizontal", color="steelblue")
    ax_right.axhline(m2_true, color="red", linestyle="-", lw=0.7)
    ax_right.axhline(m2_median, color="yellowgreen", linestyle="-", lw=0.7)
    ax_right.axhline(m2_CI_90[0], color="gray", linestyle="--", lw=0.7)
    ax_right.axhline(m2_CI_90[1], color="gray", linestyle="--", lw=0.7)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # --- Styling ---
    ax_main.set_xlabel(r"$m_1$ ($M_\odot$)")
    ax_main.set_ylabel(r"$m_2$ ($M_\odot$)")
    ax_main.legend(loc="upper right")
    ax_main.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()

    return None