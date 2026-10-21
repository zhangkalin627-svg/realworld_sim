"""Coil inductance (SI in, nH in the report). Simulation/analytic results only, never measurements.

Shapes:
 - single-layer solenoid: Nagaoka (exact current sheet) and Wheeler (approx)
 - multilayer solenoid: Wheeler
 - flat spiral (pancake): Wheeler
 - single circular loop of round wire: thin-wire closed form
All inputs are geometry arguments; nothing is hard-coded.
"""
import math
import numpy as np
from scipy.special import ellipk, ellipe
MU0 = 4e-7 * math.pi
IN = 0.0254

def nagaoka_K(length, radius):
    """Nagaoka coefficient K for a current sheet of given length and radius (r = coil radius).
    L = K * MU0 * N^2 * pi r^2 / length. K -> 1 for length >> radius."""
    x = length / (2 * radius)        # l/(2r)
    k2 = 1.0 / (1.0 + x * x)         # k^2
    k = math.sqrt(k2); kp2 = 1 - k2; kp = math.sqrt(kp2)
    Kk, Ek = ellipk(k2), ellipe(k2)
    return (4.0 / (3 * math.pi * kp)) * ((kp2 / k2) * (Kk - Ek) + Ek - k)

def solenoid_long(N, radius, length):
    """Infinite-solenoid approximation: MU0 N^2 pi r^2 / l."""
    return MU0 * N**2 * math.pi * radius**2 / length

def solenoid_nagaoka(N, radius, length):
    """Single-layer solenoid as a uniform current sheet (radius = mean winding radius)."""
    return nagaoka_K(length, radius) * solenoid_long(N, radius, length)

def solenoid_wheeler(N, radius, length):
    """Wheeler 1928 single-layer formula, SI form: MU0 N^2 pi r^2 /(l + 0.9 r) (stated accuracy ~1% for l > 0.8 r)."""
    return MU0 * N**2 * math.pi * radius**2 / (length + 0.9 * radius)

def multilayer_wheeler(N, mean_radius, length, build):
    """Wheeler multilayer solenoid: L[uH] = 0.8 a^2 N^2 /(6a + 9l + 10b), inches; a=mean radius, l=length, b=winding build (radial thickness)."""
    a, l, b = mean_radius / IN, length / IN, build / IN
    return 0.8 * a * a * N * N / (6 * a + 9 * l + 10 * b) * 1e-6

def spiral_wheeler(N, mean_radius, width):
    """Wheeler flat spiral: L[uH] = N^2 a^2 /(8a + 11c), inches; a=mean radius, c=winding width (outer-inner radius)."""
    a, c = mean_radius / IN, width / IN
    return N * N * a * a / (8 * a + 11 * c) * 1e-6

def loop_thin_wire(radius, wire_radius):
    """Single circular loop of round wire (high-frequency current on surface): MU0 R (ln(8R/a) - 2)."""
    return MU0 * radius * (math.log(8 * radius / wire_radius) - 2)


NH = 1e9

def filament_self(radius, wire_radius, current="surface"):
    """Self inductance of one circular turn of round wire. surface: HF skin current (Y=0); uniform: DC (Y=1/4)."""
    y = 0.0 if current == "surface" else 0.25
    return MU0 * radius * (math.log(8 * radius / wire_radius) - 2 + y)

def coil_filament(N, radius, pitch, wire_radius, current="surface"):
    """Few-turn single-layer coil as N exact coaxial circular filaments spaced by `pitch`
    (L = sum self + sum 2*M, M exact elliptic). Right model for 1..10 turns where the
    continuous-sheet Nagaoka/Wheeler formulas are not valid. Returns henries."""
    L = N * filament_self(radius, wire_radius, current)
    for i in range(N):
        for j in range(i + 1, N):
            L += 2 * mutual_coaxial_loops(radius, radius, (j - i) * pitch)
    return L

def two_wire_lead(length, spacing, wire_radius):
    """Go-and-return pair of round parallel leads (length >> spacing): MU0 l/pi * (ln(spacing/a) - 1 + a/spacing...) ~ acosh form.
    Uses MU0 l /pi * acosh(spacing/(2a)) (exact line inductance per length, end effects ignored)."""
    return MU0 * length / math.pi * math.acosh(spacing / (2 * wire_radius))

def busbar_pair(length, width, gap):
    """Parallel-plate busbar pair, MU0 l gap / width (valid width >> gap, fringing ignored -> underestimate)."""
    return MU0 * length * gap / width

def mutual_coaxial_loops(r1, r2, d):
    if d == 0: raise ValueError("coincident loops")
    k2 = 4 * r1 * r2 / ((r1 + r2) ** 2 + d * d)
    k = math.sqrt(k2)
    return MU0 * math.sqrt(r1 * r2) * ((2 / k - k) * ellipk(k2) - 2 / k * ellipe(k2))

def coil_inductance(N, radius, wire_radius, pitch=None, length=None, lead=None, current="surface"):
    """Report in nH. radius: mean turn radius [m]; wire_radius: conductor radius (tube outer radius) [m];
    pitch: turn-to-turn spacing [m] (or give length = (N-1)*pitch + 2*wire_radius).
    lead: optional dict(kind='two_wire', length=, spacing=, wire_radius=) or dict(kind='busbar', length=, width=, gap=).
    Returns dict of terms (nH) + flags. Simulation/analytic estimate, no measurement."""
    if pitch is None:
        pitch = (length - 2 * wire_radius) / max(N - 1, 1) if length else 2.2 * wire_radius
    out = {"unit": "nH", "N": N}
    out["coil_filament"] = coil_filament(N, radius, pitch, wire_radius, current) * NH
    L_len = (N - 1) * pitch + 2 * wire_radius
    out["coil_nagaoka_sheet"] = solenoid_nagaoka(N, radius, L_len) * NH
    out["coil_wheeler_sheet"] = solenoid_wheeler(N, radius, L_len) * NH
    if lead:
        k = lead["kind"]
        out["lead_busbar"] = (two_wire_lead(lead["length"], lead["spacing"], lead["wire_radius"]) if k == "two_wire"
                              else busbar_pair(lead["length"], lead["width"], lead["gap"])) * NH
    else:
        out["lead_busbar"] = None
    out["terminal_end_effect"] = "NOT MODELLED"
    out["total_modelled"] = out["coil_filament"] + (out["lead_busbar"] or 0.0)
    w = []
    if N < 10 or L_len < 1.6 * radius:
        w.append("few-turn/short coil: sheet formulas (Nagaoka/Wheeler) are NOT reliable here; use coil_filament")
    if lead is None: w.append("leads/busbar not included in total")
    w.append("terminal/joint/end connection inductance not modelled; at nH level these can be comparable to the coil")
    w.append("workpiece/coupling (L drop with load) not included; air-core value only")
    out["warnings"] = w
    return out

# ---- in-circuit effective L from resonance (helper; topology NOT assumed) ----
def L_from_resonance(f, C):
    """Ideal lossless LC: L = 1/((2 pi f)^2 C). Valid as written for a series LC and for an ideal parallel LC.
    A parallel tank with coil resistance R resonates (max |Z|) near w0^2 = 1/(L C) - (R/L)^2, a loaded series tank
    reads the same w0 only when R is small. Use together with the user's own formula before comparing."""
    return 1.0 / ((2 * math.pi * f) ** 2 * C)

def f_from_LC(L, C):
    """Resonant frequency of ideal LC (Hz)."""
    return 1.0 / (2 * math.pi * math.sqrt(L * C))

def frequency_table(L_values_H, C_values_F):
    """List of (L_nH, C_uF, f_kHz) for ideal LC, for comparing predicted L with machine-tool readings."""
    return [(L * 1e9, C * 1e6, f_from_LC(L, C) / 1e3) for L in L_values_H for C in C_values_F]

# ---- loaded coil (long-coil analytic, workpiece = solid cylinder in an infinite solenoid) ----
def loaded_long_coil_per_length(N_per_m, coil_radius, wp_radius, f, sigma, mu_r=1.0):
    """Impedance per unit length Z/(jw) = MU0 n^2 (pi (R^2-a^2) + pi a^2 * fac), fac from Bessel ratio.
    Returns (L_per_m [H/m], R_per_m [ohm/m]). 1D infinite-coil model: no ends, no leads; use only for l >> 2R."""
    from scipy.special import ive
    w = 2 * math.pi * f
    g = np.sqrt(1j * w * sigma * MU0 * mu_r) * wp_radius
    fac = complex(mu_r) if abs(g) < 1e-6 else complex(mu_r * (2 / g) * ive(1, g) / ive(0, g))
    Zjw = MU0 * N_per_m ** 2 * (math.pi * (coil_radius ** 2 - wp_radius ** 2) + math.pi * wp_radius ** 2 * fac)
    return Zjw.real, -w * Zjw.imag

def rect_section_ring(N, radius, radial, axial, nr=10, nz=20):
    """Single-layer coil of N turns, each turn's current uniform over a rectangular section
    (radial x axial). Exact superposition of nr*nz coaxial filaments (elliptic-integral mutuals, GMD self term
    0.2235*(dr+dz)); all N turns assumed stacked at the same place, so use for N=1 or as a thick-band model.
    radius = mean radius of the section. Returns henries."""
    dr, dz = radial / nr, axial / nz
    r = radius - radial / 2 + (np.arange(nr) + 0.5) * dr
    z = -axial / 2 + (np.arange(nz) + 0.5) * dz
    R, Z = np.meshgrid(r, z); R = R.ravel(); Z = Z.ravel(); n = len(R); w = N / n
    r1 = R[:, None]; r2 = R[None, :]; d = np.abs(Z[:, None] - Z[None, :])
    k2 = 4 * r1 * r2 / ((r1 + r2) ** 2 + d * d)
    with np.errstate(all="ignore"):
        k = np.sqrt(k2)
        M = MU0 * np.sqrt(r1 * r2) * ((2 / k - k) * ellipk(k2) - 2 / k * ellipe(k2))
    gmd = 0.2235 * (dr + dz)
    M[np.diag_indices(n)] = MU0 * R * (np.log(8 * R / gmd) - 2)
    return w * w * M.sum()

# ---- active-coil estimate (industry training-formula style) ----
def penetration_depth(rho, f, mu_r=1.0):
    """Current penetration (skin) depth [m]: sqrt(2 rho / (w MU0 mu_r)). rho [ohm m], f [Hz]."""
    w = 2 * math.pi * f
    return math.sqrt(2 * rho / (w * MU0 * mu_r))

def active_area_gap(d_coil_id, d_workpiece):
    """Annular cross-section between coil bore and workpiece [m^2]."""
    return math.pi / 4 * (d_coil_id ** 2 - d_workpiece ** 2)

def active_area_workpiece(d_workpiece, delta):
    """Skin annulus inside the workpiece where the field exists [m^2]; capped at the full section."""
    r = d_workpiece / 2
    d_eff = min(delta, r)
    return math.pi * (r ** 2 - (r - d_eff) ** 2)

def active_coil_inductance(N, length, area_m, mu_r=1.0):
    """L = MU0 mu_r N^2 A_m / l_coil (henries). Long-coil approximation with magnetic area A_m:
    no end-effect correction, so it reads high for short or few-turn coils (compare solenoid_nagaoka
    and coil_filament before using)."""
    return MU0 * mu_r * N ** 2 * area_m / length

def loaded_active_coil(N, length, d_coil_id, d_workpiece, rho_w, mu_r_w, f):
    """Loaded active-coil estimate: A_m = gap annulus + mu_r_w-weighted workpiece skin annulus
    (skin depth from penetration_depth). Long-coil model, no ends/leads; magnetic workpieces
    (mu_r_w >> 1) make the inside area significant. Returns dict; simulation estimate only."""
    a_gap = active_area_gap(d_coil_id, d_workpiece)
    delta = penetration_depth(rho_w, f, mu_r_w)
    a_wp = active_area_workpiece(d_workpiece, delta)
    L_gap = active_coil_inductance(N, length, a_gap)
    L_wp = active_coil_inductance(N, length, a_wp, mu_r_w)
    return {"unit": "nH", "L_total": (L_gap + L_wp) * NH, "L_gap": L_gap * NH,
            "L_workpiece": L_wp * NH, "A_gap_mm2": a_gap * 1e6, "A_wp_mm2": a_wp * 1e6,
            "skin_depth_mm": delta * 1e3,
            "warnings": ["long-coil approximation: no end effects; high for short/few-turn coils",
                         "linear mu_r assumed; saturated magnetic workpieces read lower in practice",
                         "leads/busbar/terminals not included"]}
