import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime, root

x = np.linspace(-3, 3, 40)
y = np.linspace(-3, 3, 40)


def func(z):
    return np.array([-z[0]**2 + z[1]**2, -z[0]**2 - z[1]**2 + 1])

X, Y = np.meshgrid(x, y)
u, v = func([X, Y]) 

def analytic_jacobian(pt):
    x0, y0 = pt
    return np.array([[-2*x0,  2*y0],
                     [-2*x0, -2*y0]])

def numeric_jacobian(pt, eps=None):
    if eps is None:
        eps = np.sqrt(np.finfo(float).eps)
    
    rows = [approx_fprime(np.asarray(pt), lambda z, i=i: func(z)[i], eps) for i in (0, 1)]
    return np.vstack(rows) 


fig, ax = plt.subplots(figsize=(8, 6))
ax.quiver(X, Y, u, v, scale=50, pivot='mid', color='teal')

def _norm(v):
    return v / (np.linalg.norm(v) + 1e-16)


grid_scale = 0.6 * (x[-1] - x[0]) / 10.0

def find_equilibria_numerical(func, guesses, tol=1e-8):
    found = []
    for g in guesses:
        sol = root(lambda z: func(z), g)
        if not sol.success:
            continue
        z = sol.x
        
        if np.linalg.norm(func(z)) > 1e-6:
            continue
        # deduplicate (real/complex-aware)
        dup = False
        for existing in found:
            if np.linalg.norm(np.asarray(existing['pt']).real - z) < tol:
                dup = True
                break
        if not dup:
            found.append({'pt': z, 'success': True, 'msg': sol.message})
    return found

# coarse initial guesses across domain (can increase density if needed)
init_x = np.linspace(-2.5, 2.5, 9)
init_y = np.linspace(-2.5, 2.5, 9)
inits = [np.array([xx, yy]) for xx in init_x for yy in init_y]

num_eqs = find_equilibria_numerical(func, inits)

numeric_info = []
for item in num_eqs:
    z = np.asarray(item['pt'])
    # check for non-negligible imaginary parts (shouldn't usually happen for real-valued func)
    if np.any(np.abs(np.imag(z)) > 1e-8):
        numeric_info.append({'pt': z, 'complex': True, 'jac': None, 'eigs': None, 'eigvecs': None})
        continue
    z = z.real
    Jnum = numeric_jacobian(z)
    eigs, eigvecs = np.linalg.eig(Jnum)
    numeric_info.append({'pt': z, 'complex': False, 'jac': Jnum, 'eigs': eigs, 'eigvecs': eigvecs})

for info in numeric_info:
    if info['complex']:
        print(f"numerical equilibrium (complex) at {info['pt']}")
    else:
        print(f"numerical equilibrium at {info['pt']}: eigs={info['eigs']}, eigvecs={info['eigvecs'].T}")


real_pts = [info for info in numeric_info if not info['complex']]
complex_pts = [info for info in numeric_info if info['complex']]

if real_pts:
    rp = np.vstack([info['pt'] for info in real_pts])
    ax.scatter(rp[:, 0], rp[:, 1], marker='*', color='green', s=80, zorder=6, label='numeric equilibria (real)')
if complex_pts:
    cp = np.vstack([info['pt'].real for info in complex_pts])
    ax.scatter(cp[:, 0], cp[:, 1], marker='o', facecolors='none', edgecolors='orange', s=80,
               zorder=6, label='numeric equilibria (complex proj.)')

for info in real_pts:
    px, py = info['pt']
    eigs = info['eigs']
    eigvecs = info['eigvecs']
    for lam, vec in zip(eigs, eigvecs.T):
        if np.isclose(lam.imag, 0.0, atol=1e-8):
            v = _norm(vec.real)
            color = 'blue' if lam.real < 0 else 'red'
            ax.quiver(px, py, v[0]*grid_scale, v[1]*grid_scale,
                      angles='xy', scale_units='xy', scale=1,
                      color=color, width=0.025, zorder=7)
        else:
            vr = _norm(vec.real) * grid_scale
            vi = _norm(vec.imag) * grid_scale
            ax.plot([px, px+vr[0]], [py, py+vr[1]], '-', color='purple', linewidth=2, zorder=7)
            ax.plot([px, px+vi[0]], [py, py+vi[1]], '--', color='purple', linewidth=1.5, zorder=7)


handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.set_aspect('equal')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.legend(by_label.values(), by_label.keys(), loc='upper right')
plt.show()

