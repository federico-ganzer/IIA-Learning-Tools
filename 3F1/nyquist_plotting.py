"""
Nyquist plot for a discrete-time transfer function G(z).

- Evaluates G(z) along the unit circle: z = exp(j*omega), omega in [0, 2*pi).
- Two ways to define G(z):
    (A) As a direct lambda in z
    (B) From zeros, poles, gain using a helper builder

- Handles poles close to / on the unit circle by masking those points
  so the plot does not jump across asymptotes.
- Chooses axis limits robustly using percentiles (outlier-resistant).
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1) Helper: build G(z) from P/Z/K
# ------------------------------
def build_G_from_pzk(zeros=None, poles=None, k=1.0):
    """
    Creates a callable G(z) along with a callable denom(z) for masking.

    zeros, poles: lists (or arrays) of complex numbers for z-plane zeros/poles
    k:            real/complex gain

    Returns:
        G(z): function mapping complex array z -> complex array
        denom(z): function that evaluates the denominator only (for masking)
    """
    zeros = [] if zeros is None else list(zeros)
    poles = [] if poles is None else list(poles)

    def poly_from_roots(roots, z):
        # Evaluate product (z - r) over all r in roots
        out = np.ones_like(z, dtype=complex)
        for r in roots:
            out = out * (z - r)
        return out

    def numerator(z):
        if len(zeros) == 0:
            return np.ones_like(z, dtype=complex)
        return poly_from_roots(zeros, z)

    def denominator(z):
        if len(poles) == 0:
            return np.ones_like(z, dtype=complex)
        return poly_from_roots(poles, z)

    def G(z):
        return k * numerator(z) / (denominator(z) + 1e-9)  # small offset to avoid div by zero

    return G, denominator


# ------------------------------
# 2) Example definitions of G(z)
# ------------------------------
# (A) Direct lambda definition in z (edit this freely)
#     Example: G(z) = 0.5 * (z + 0.2) / (z*(z - 0.8))
# G = lambda z: 0.5 * (z + 0.2) / (z * (z - 0.8))
# denom = None # we don't have direct access to the denominator here

# (B) Poles/zeros/gain definition (uncomment and edit to use)
#     Example: zeros at z = -0.2; poles at z = 0 (origin) and 0.8; gain 0.5
G, denom = build_G_from_pzk(
    zeros=[-2],
    poles=[ 1/3, -1/2],  # pole near unit circle at 45 degrees
    k=-1/6,
)

# If you need to test a pole on the unit circle (e.g., at z = 1),
# you can do poles=[1.0]. The mask below will remove points near it
# so the plot remains readable (axis limits also stay reasonable).


# ------------------------------
# 3) Plot settings
# ------------------------------
N_omega = 4000          # number of frequency samples on [0, 2*pi)
omega = np.linspace(0.0, 2.0*np.pi, N_omega, endpoint=False)
z = np.exp(1j * omega)  # unit circle

# ------------------------------
# 4) Evaluate and mask near poles
# ------------------------------
# We try to avoid plotting through singularities by detecting small denominators.
# When denom(z) is available (p/z/k definition), we mask points where |denom| is small.
# Otherwise, as a fallback heuristic, we mask points where |G| is extremely large.

# Compute raw response
G_vals = G(z)

mask = np.ones_like(G_vals, dtype=bool)  # True means "keep"

if denom is not None:
    den_vals = denom(z)
    # Relative threshold: treat very small |den| as a pole vicinity.
    # The scale below uses a fraction of the median magnitude to be scale-aware.
    # If the median is 0 (can happen in edge cases), fall back to an absolute tiny threshold.
    den_mag = np.abs(den_vals)
    scale = np.median(den_mag)
    if scale == 0.0:
        scale = 1.0
    pole_tol = 1e-6 * scale
    near_pole = den_mag < pole_tol
    mask = ~near_pole

else:
    # Heuristic based on |G| magnitude: mask the largest few percent
    G_mag = np.abs(G_vals)
    cutoff = np.percentile(G_mag[np.isfinite(G_mag)], 99.5)
    # Anything way above that is likely crossing an asymptote
    near_pole = G_mag > 5.0 * cutoff
    mask = ~near_pole

# Insert NaNs where masked to break the plot line across asymptotes
G_plot = G_vals.copy()
G_plot[~mask] = np.nan

# ------------------------------
# 5) Robust axis limits (percentile-based)
# ------------------------------
# This keeps outliers from wrecking the view, while still showing most of the curve.
re = np.real(G_plot)
im = np.imag(G_plot)
finite = np.isfinite(re) & np.isfinite(im)

if np.any(finite):
    re_f = re[finite]
    im_f = im[finite]
    # Use central 98% of the data for limits
    rmin, rmax = np.percentile(re_f, [1.0, 99.0])
    imin, imax = np.percentile(im_f, [1.0, 99.0])

    # Add a small margin
    def pad(a, b, frac=0.05, min_pad=1e-3):
        span = max(b - a, min_pad)
        pad_amt = frac * span
        return a - pad_amt, b + pad_amt

    xlim = pad(rmin, rmax)
    ylim = pad(imin, imax)
else:
    # Fallback limits
    xlim = (-1.0, 1.0)
    ylim = (-1.0, 1.0)

# ------------------------------
# 6) Plot
# ------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(np.real(G_plot), np.imag(G_plot), lw=1.5, label=r"$G(e^{j\theta}), \theta \in [0, 2\pi)$")
ax.axhline(0.0, color="gray", lw=0.8)
ax.axvline(0.0, color="gray", lw=0.8)

ax.set_xlabel(r"$Re\{G(e^{j\theta})\}$")
ax.set_ylabel(r"$Im\{G(e^{j\theta})\}$")
ax.set_title("Discrete-Time Nyquist Plot")
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect("equal", adjustable="box")
ax.grid(True, ls=":", lw=0.6)
ax.legend(loc="best")

plt.tight_layout()
plt.show()


# ------------------------------
# 7) Tips for adjustment
# ------------------------------
# - To change the transfer function, edit either:
#       G_direct = lambda z: ...
#   or switch to the p/z/k builder by setting use_pzk = True and editing
#   the zeros, poles, and gain in the call to build_G_from_pzk(...).
#
# - If you have *exactly* a pole on the unit circle, the mask above will
#   remove samples whose denominator is ~0 so the curve does not "shoot off."
#   You can make the masking more or less aggressive by changing 'pole_tol'
#   or the percentile cut used in the magnitude heuristic.
#
# - If the plot still looks cramped or too zoomed-out, tweak the percentile
#   values in the axis limit section (e.g., use [2, 98] instead of [1, 99]).
#
# - Increase N_omega for a smoother curve (at the cost of more computation).

# ------------------------------
# 6) Plot + transformed view
# ------------------------------

def softclip_to_circle(G_raw, R):
    """
    Phase-preserving radial soft-clip onto radius R.

    H(G) = (R * tanh(|G|/R)) * exp(j*angle(G))
         = G * (R * tanh(|G|/R) / |G|)   with |G|>0
    Properties:
      - For |G| << R: tanh(|G|/R) ~ |G|/R  -> H(G) ~ G (identity)
      - For |G| -> inf: tanh(|G|/R) -> 1  -> |H(G)| -> R with same phase
    """
    G = np.asarray(G_raw, dtype=complex)

    # Magnitude and phase; handle non-finite magnitudes and phases robustly.
    mag = np.abs(G)
    ang = np.angle(G)

    # Some samples can be inf/nan near poles. We want a continuous angle to
    # keep the encirclement structure. We unwrap where finite, then interpolate.
    finite_idx = np.isfinite(mag) & np.isfinite(ang)
    ang_unwrapped = np.empty_like(ang)
    if np.any(finite_idx):
        # Unwrap on finite subset and linearly interpolate over gaps
        ang_f = np.unwrap(ang[finite_idx])
        x = np.flatnonzero(finite_idx)
        xp = x
        fp = ang_f
        # Build full array by interpolation; for edges, hold nearest value
        ang_interp = np.interp(np.arange(len(G)), xp, fp)
        # If everything was finite, ang_interp equals unwrapped; else it fills gaps
        ang_unwrapped[:] = ang_interp
    else:
        # Fallback: all invalid, set zero phase
        ang_unwrapped[:] = 0.0

    # Replace non-finite magnitudes by a very large value so tanh -> 1
    mag_safe = mag.copy()
    mag_safe[~np.isfinite(mag_safe)] = 1e12

    # Radial soft-clip
    rad = R * np.tanh(mag_safe / R)          # in [0, R)
    H = rad * np.exp(1j * ang_unwrapped)
    return H


# Choose R automatically so "normal" values are unaffected but asymptotes clip.
# We use a robust percentile on the unmasked response; increase or set manually if needed.
G_mag_all = np.abs(G_vals)
finite_all = np.isfinite(G_mag_all)
if np.any(finite_all):
    # Typical scale of the response, excluding extreme outliers
    R_auto = np.percentile(G_mag_all[finite_all], 95.0)
    # Ensure R is not tiny (helps when most of the curve lives very near 0)
    R_clip = max(R_auto, 1.0)
else:
    R_clip = 1.0

# Compute transformed curve on the raw (unmasked) values so we keep continuity
H_vals = softclip_to_circle(G_vals, R_clip)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# ---- Left: original Nyquist (masked to avoid spikes) ----
ax1.plot(np.real(G_plot), np.imag(G_plot), lw=1.5, label=r"$G(e^{j \theta})$")
ax1.axhline(0.0, color="gray", lw=0.8)
ax1.axvline(0.0, color="gray", lw=0.8)
ax1.plot([-1.0], [0.0], marker="x", markersize=8, color="black")
ax1.text(-1.0, 0.0, "  -1", va="center", ha="left")
ax1.set_xlabel(r"$Re\{G(e^{j \theta})\}$")
ax1.set_ylabel(r"$Im\{G(e^{j \theta})\}$")
ax1.set_title("Nyquist (original)")
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.set_aspect("equal", adjustable="box")
ax1.grid(True, ls=":", lw=0.6)
ax1.legend(loc="best")

# ---- Right: transformed Nyquist with infinities mapped to circle of radius R_clip ----
ax2.plot(np.real(H_vals), np.imag(H_vals), lw=1.5, label=r"$H(G(e^{j \theta}))$")
ax2.axhline(0.0, color="gray", lw=0.8)
ax2.axvline(0.0, color="gray", lw=0.8)

# Draw the clipping circle |w| = R_clip (where infinities land)
theta_draw = np.linspace(0.0, 2.0*np.pi, 512, endpoint=True)
ax2.plot(R_clip*np.cos(theta_draw), R_clip*np.sin(theta_draw),
         ls="--", lw=1.0, label=r"$|w| = R_{\text{clip}}$")

ax2.set_xlabel(r"$Re\{H(G)\}$")
ax2.set_ylabel(r"$Im\{H(G)\}$")
ax2.set_title("Nyquist with infinity mapped to circle")

# Axes: include the circle and most of the transformed curve.
Hx, Hy = np.real(H_vals), np.imag(H_vals)
finiteH = np.isfinite(Hx) & np.isfinite(Hy)
if np.any(finiteH):
    # Use central 98% plus ensure circle fully visible
    rx = np.percentile(Hx[finiteH], [1.0, 99.0])
    ry = np.percentile(Hy[finiteH], [1.0, 99.0])

    def pad2(a, b, frac=0.05, min_pad=1e-3):
        span = max(b - a, min_pad)
        pad_amt = frac * span
        return a - pad_amt, b + pad_amt

    xlim_t = pad2(rx[0], rx[1])
    ylim_t = pad2(ry[0], ry[1])

    # Make sure the full clipping circle is inside the frame
    rmax = R_clip * 1.05
    xlim_t = (min(xlim_t[0], -rmax), max(xlim_t[1], rmax))
    ylim_t = (min(ylim_t[0], -rmax), max(ylim_t[1], rmax))
else:
    rmax = R_clip * 1.05
    xlim_t = (-rmax, rmax)
    ylim_t = (-rmax, rmax)

ax2.set_xlim(xlim_t)
ax2.set_ylim(ylim_t)
ax2.set_aspect("equal", adjustable="box")
ax2.grid(True, ls=":", lw=0.6)
ax2.legend(loc="best")

plt.tight_layout()
plt.show()

# ------------------------------
# Notes:
# - R_clip is chosen automatically from the 95th percentile of |G| so
#   typical magnitudes are nearly unchanged while pole blow-ups map to
#   the circle. If you want a different threshold, set R_clip manually.
# - The soft clip preserves phase and compresses only the radius, so
#   encirclements remain easy to see without spikes to infinity.