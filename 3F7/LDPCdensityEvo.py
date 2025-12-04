import numpy as np
import matplotlib.pyplot as plt

def construct_poly(x, p):
    degrees = np.arange(1, len(p) + 1)
    probs = p / np.sum(p)
    return np.sum(probs * x**(degrees - 1))

def _density_evolution_regular_codes(epsilon, dv, dc, max_iter=100, tolerance=1e-12):
    """
    Compute error probability evolution for regular LDPC on BEC.
    
    Recursion: p_t = ε(1 - (1 - p_{t-1})^{d_c-1})^{d_v-1}
    
    Args:
        epsilon: Erasure probability
        dv: Variable degree
        dc: Check degree
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
    
    Returns:
        history: List of dicts with {t, p, delta}
    """
    p = epsilon
    history = [{'t': 0, 'p': p, 'delta': 0}]
    
    for t in range(1, max_iter + 1):
        p_next = epsilon * (1 - (1 - p)**(dc - 1))**(dv - 1)
        delta = abs(p_next - p)
        history.append({'t': t, 'p': p_next, 'delta': delta})
        p = p_next
        
        if delta < tolerance:
            break
    
    return history

    return history[-1]['p'] < threshold

def is_converged(history, threshold=1e-10):
    final_p = history[-1]['p']
    return final_p < threshold

def find_threshold_bisection(lambda_poly, rho_poly, max_iter=1000, precision=1e-12):
    """
    Find capacity threshold using binary search.
    
    Returns the maximum ε for which p_t → 0.
    """
    left, right = 0, 1 - 1e-6
    best_threshold = 0
    
    for _ in range(max_iter):
        mid = (left + right) / 2
        history = density_evolution(mid, lambda_poly, rho_poly, max_iter)
        
        if is_converged(history):
            best_threshold = mid
            left = mid
        else:
            right = mid
        
        if right - left < precision:
            break
    
    return best_threshold

def density_evolution(eps, lambda_poly, rho_poly, max_iter= 1000, tol= 1e-10):
    x = eps
    hist = [x]
    for _ in range(max_iter):
        inner = 1 -construct_poly(1 - x, rho_poly)
        x_next = eps * construct_poly(inner, lambda_poly)
        hist.append(x_next)
        if abs(x_next - x) < tol:
            break
        x = x_next
    return np.array(hist)

# dv, dc = 3, 6 Generalise ie. create a polynomial constructor for irregular codes
max_iter = 100

lamda_poly = [0, 0.5, 0.5]  # lamda(x) = 0.5x + 0.5x^2
rho_poly = [0, 0, 1.0]      # rho(x) = x^2
dv = sum(i * p for i, p in enumerate(lamda_poly))  # Average variable degree
dc = sum(i * p for i, p in enumerate(rho_poly))   # Average check degree

threshold = find_threshold_bisection(lamda_poly, rho_poly)
print(f"Estimated threshold: {threshold:.6f}")
print(f"Shannon limit: {1 - dv/dc:.4f}")
print()

epsilons = [0.425, 0.426, 0.427, 0.43]
fig, ax = plt.subplots(figsize=(10, 6))

for eps in epsilons:
    history = density_evolution(eps, lamda_poly, rho_poly, max_iter)
    ts = [h['t'] for h in history]
    ps = [max(h['p'], 0) for h in history]   # avoid zeros on log scale
    ax.plot(ts, ps, marker='o', label=f'ε = {eps}', linewidth=2)

ax.set_xlabel('Iteration t', fontsize=12)
ax.set_ylabel('Error probability p_t', fontsize=12)
ax.set_title(f'Density Evolution: (d_v={dv}, d_c={dc})', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


    