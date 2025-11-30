import numpy as np
import matplotlib.pyplot as plt

def density_evolution(epsilon, dv, dc, max_iter=100, tolerance=1e-12):
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

def is_converged(history, threshold=1e-3):
    return history[-1]['p'] < threshold

def find_threshold_bisection(dv, dc, max_iter=1000, precision=1e-12):
    """
    Find capacity threshold using binary search.
    
    Returns the maximum ε for which p_t → 0.
    """
    left, right = 0, 1 - 1e-6
    best_threshold = 0
    
    for _ in range(max_iter):
        mid = (left + right) / 2
        history = density_evolution(mid, dv, dc, max_iter)
        
        if is_converged(history):
            best_threshold = mid
            left = mid
        else:
            right = mid
        
        if right - left < precision:
            break
    
    return best_threshold

dv, dc = 3, 6
max_iter = 100

threshold = find_threshold_bisection(dv, dc)
print(f"Estimated threshold: {threshold:.6f}")
print(f"Shannon limit: {1 - dv/dc:.4f}")
print()

epsilons = [0.40, 0.41, 0.42, 0.43]
fig, ax = plt.subplots(figsize=(10, 6))

for eps in epsilons:
    history = density_evolution(eps, dv, dc, max_iter)
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


    