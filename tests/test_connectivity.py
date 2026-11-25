"""Tests of connectivity.py: the connectivity numbers of the paper (Methods "Connectivity", Extended Data figure)
and the pipeline that measures the connectivity of one region.

Run from the repository root:
    python -m pytest              fast tests
    python -m pytest -m slow      also the tests that run DisPerSE (set DISPERSE_BIN to the folder of mse/skelconv)

Every test states the hand computation it checks.  Conventions used throughout:
  * comoving h^-1 Mpc; the analysis cube is centred on the halo, +-15 h^-1 Mpc, grid cell 0.125 h^-1 Mpc (240^3);
  * cloud-in-cell (CIC) deposit onto the grid nodes lo + i*cell; grid index [i, j, k] <-> (x, y, z);
  * DisPerSE writes a point in FITS pixel order (k, j, i); it maps to (x, y, z) = -lhalf + (i, j, k) * cell;
  * kappa = number of filaments with at least one point inside the sphere (distance <= radius) and one outside.
"""
import json
import os
import shutil
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import connectivity as C  # noqa: E402

DATA = ROOT / "data" / "connectivity_cluster129.hdf5"
RHO = (0.307115 - 0.048) * 2.775e11          # mean DM density, h^2 Msun Mpc^-3 (comoving)
API_ERRORS = (AttributeError, NameError, ImportError, TypeError)


# ---- helpers ---------------------------------------------------------------------------------------------------------

def segment(p0, p1, n=20):
    """Straight polyline from p0 to p1 with n evenly spaced points."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return p0 + np.linspace(0.0, 1.0, n)[:, None] * (p1 - p0)


def length(a):
    return float(np.linalg.norm(np.diff(np.asarray(a, float), axis=0), axis=1).sum())


def raises_error(fn):
    """fn() must raise an error of its own (not a missing function or a wrong call)."""
    with pytest.raises(Exception) as e:
        fn()
    assert not isinstance(e.value, API_ERRORS), repr(e.value)
    return e.value


def mergeable(types):
    """Flat 2N bool array (start of arc i at 2i, end at 2i+1): True at saddles (type 2)."""
    return np.asarray(types, int).ravel() == 2


def write_ndskl(path, cps, arcs, bbox=240):
    """Write a skeleton in the NDskl_ascii format of `skelconv -to NDskl_ascii`.

    cps:  list of (type, position in file pixel order (k, j, i), boundary flag)
    arcs: list of (cp1, cp2, sampling points in file pixel order (n, 3)), running from cp1 to cp2.
    """
    cps = [(int(t), np.asarray(p, float), int(b)) for t, p, b in cps]
    arcs = [(int(c1), int(c2), np.asarray(pts, float)) for c1, c2, pts in arcs]
    links = [[] for _ in cps]
    for f, (c1, c2, pts) in enumerate(arcs):
        assert np.allclose(pts[0], cps[c1][1]) and np.allclose(pts[-1], cps[c2][1]), "test file: arc ends != its CPs"
        links[c1].append((c2, f))
        links[c2].append((c1, f))
    fmt = lambda p: " ".join(f"{v:.6f}" for v in p)        # noqa: E731
    L = ["ANDSKEL", "3", "#No comments", f"BBOX [0,0,0] [{bbox},{bbox},{bbox}]", "[CRITICAL POINTS]", str(len(cps))]
    for n, (t, p, b) in enumerate(cps):
        L.append(f"{t} {fmt(p)} {1.0 + n:.6f} -1 {b}")
        L.append(f" {len(links[n])}")
        L += [f" {d} {f}" for d, f in links[n]]
    L += ["[FILAMENTS]", str(len(arcs))]
    for c1, c2, pts in arcs:
        L.append(f"{c1} {c2} {len(pts)}")
        L += [" " + fmt(q) for q in pts]
    names = ["persistence_ratio", "persistence_nsigmas", "persistence", "persistence_pair", "parent_index",
             "parent_log_index", "log_field_value", "field_value", "cell"]
    L += ["[CRITICAL POINTS DATA]", str(len(names))] + names
    L += [f"2.0 3.0 1.0 -1 -1 -1 0.0 {1.0 + n:.6f} {n}.0" for n in range(len(cps))]
    fnames = ["field_value", "orientation", "cell", "log_field_value", "type"]
    L += ["[FILAMENTS DATA]", str(len(fnames))] + fnames
    for _, _, pts in arcs:
        L += ["1.0 1 0.0 0.0 2"] * len(pts)
    Path(path).write_text("\n".join(L) + "\n")
    return str(path)


def to_xyz(p_file, lhalf=15.0, cell=0.125):
    """Hand mapping: file (k, j, i) -> (x, y, z) = -lhalf + (i, j, k) * cell."""
    return -lhalf + np.asarray(p_file, float)[..., ::-1] * cell


# ---- constants -------------------------------------------------------------------------------------------------------

def test_constants():
    """Spec P2: mean DM density (Omega_m - Omega_b) rho_crit with Omega_m = 0.307115, Omega_b = 0.048,
    rho_crit = 2.775e11 h^2 Msun Mpc^-3; cube half-width 15, cell 0.125, smoothing R_s = 0.5 h^-1 Mpc."""
    assert C.RHO_DM_MEAN == pytest.approx(RHO, rel=1e-12)
    assert (C.LHALF, C.CELL, C.RS) == (15.0, 0.125, 0.5)


# ---- CIC -------------------------------------------------------------------------------------------------------------

LO = np.array([-1.0, 2.0, 0.5])                 # asymmetric corner, multiples of the cell (exact arithmetic)
CELL, NG, MDM = 0.125, 8, 1.0e10
AMP = MDM / CELL ** 3 / RHO                     # 1 + delta of a whole particle on one node


def test_cic_particle_on_a_node():
    """A particle exactly on node (2, 3, 5) puts all its mass there: 1 + delta = m / (cell^3 rho_mean) on that node
    and delta = -1 elsewhere.  The asymmetric node index catches an axis swap (grid [i, j, k] <-> (x, y, z))."""
    g = C.cic_delta(LO + np.array([[2, 3, 5]]) * CELL, LO, NG, CELL, MDM)
    assert g.shape == (NG, NG, NG) and g.dtype == np.float64
    want = np.full((NG, NG, NG), -1.0)
    want[2, 3, 5] = AMP - 1.0
    np.testing.assert_allclose(g, want, rtol=1e-12, atol=1e-12)


def test_cic_particle_at_a_cell_centre():
    """A particle at the centre of the cell with corner node (2, 3, 5) shares its mass equally among the 8 nodes
    (2..3, 3..4, 5..6): 1 + delta = AMP / 8 on each."""
    g = C.cic_delta(LO + np.array([[2.5, 3.5, 5.5]]) * CELL, LO, NG, CELL, MDM)
    want = np.full((NG, NG, NG), -1.0)
    want[2:4, 3:5, 5:7] = AMP / 8 - 1.0
    np.testing.assert_allclose(g, want, rtol=1e-12, atol=1e-12)


def test_cic_linear_weights_along_x():
    """A particle a quarter cell above node (2, 3, 5) in x: weight 3/4 on node (2, 3, 5), 1/4 on (3, 3, 5)."""
    g = C.cic_delta(LO + np.array([[2.25, 3, 5]]) * CELL, LO, NG, CELL, MDM)
    want = np.full((NG, NG, NG), -1.0)
    want[2, 3, 5] = 0.75 * AMP - 1.0
    want[3, 3, 5] = 0.25 * AMP - 1.0
    np.testing.assert_allclose(g, want, rtol=1e-12, atol=1e-12)


def test_cic_index_ng_wraps_to_0():
    """Spec P1 (same wrap of index ng to 0): a particle half a cell above the last node in x deposits half on node
    ng - 1 and half on node 0."""
    g = C.cic_delta(LO + np.array([[NG - 0.5, 3, 5]]) * CELL, LO, NG, CELL, MDM)
    want = np.full((NG, NG, NG), -1.0)
    want[NG - 1, 3, 5] = 0.5 * AMP - 1.0
    want[0, 3, 5] = 0.5 * AMP - 1.0
    np.testing.assert_allclose(g, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_cic_conserves_mass(seed):
    """Spec P1: sum (1 + delta) rho_mean cell^3 = N m for particles strictly inside the cube."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 5000))
    pos = LO + rng.uniform(0, NG * CELL, (n, 3)) * (1 - 1e-12)
    g = C.cic_delta(pos, LO, NG, CELL, MDM)
    assert ((1 + g) * RHO * CELL ** 3).sum() == pytest.approx(n * MDM, rel=1e-10)


# ---- density field ---------------------------------------------------------------------------------------------------

def test_density_field_of_a_uniform_lattice_is_zero():
    """One particle on every node of a 16^3 grid (lhalf 1, cell 0.125, centred on (10, 20, 30)) with m = rho_mean
    cell^3: the density is the cosmic mean everywhere, delta = 0 before and after smoothing; float32."""
    centre = np.array([10.0, 20.0, 30.0])
    i = np.arange(16) * 0.125
    pos = (centre - 1.0) + np.stack(np.meshgrid(i, i, i, indexing="ij"), -1).reshape(-1, 3)
    d = C.density_field(pos, centre, RHO * 0.125 ** 3, lhalf=1.0, cell=0.125, rs=0.5)
    assert d.shape == (16, 16, 16) and d.dtype == np.float32
    np.testing.assert_allclose(d, 0.0, atol=1e-5)


def test_density_field_peak_and_smoothing_width():
    """One particle on node (11, 12, 13) of a 24^3 grid (lhalf 1.5) about centre (10, 20, 30); R_s = 0.25 = 2 cells.
    Hand computation: unsmoothed 1 + delta = A = m / (cell^3 rho_mean) on that node; a Gaussian of sigma = R_s / cell
    = 2 cells (truncated at 4 sigma, so it does not reach the faces) gives delta = -1 + A w0^3 at the node, with
    w0 = 1 / sum_{k=-8..8} exp(-k^2 / 8).  The peak index also checks the axis order and the cube offset."""
    centre = np.array([10.0, 20.0, 30.0])
    node = np.array([11, 12, 13])
    m = 1.0e9
    pos = (centre - 1.5 + node * 0.125)[None, :]
    d = C.density_field(pos, centre, m, lhalf=1.5, cell=0.125, rs=0.25)
    assert d.shape == (24, 24, 24) and d.dtype == np.float32
    assert np.unravel_index(np.argmax(d), d.shape) == tuple(node)
    k = np.arange(-8, 9)
    w0 = 1.0 / np.exp(-k ** 2 / 8.0).sum()
    A = m / 0.125 ** 3 / RHO
    assert float(d[tuple(node)]) == pytest.approx(-1.0 + A * w0 ** 3, rel=1e-5)


def test_density_field_default_grid():
    """Spec P2: the production grid is 240^3 (+-15 h^-1 Mpc, cell 0.125), float32."""
    centre = np.array([500.0, 480.0, 510.0])
    d = C.density_field(centre[None, :] + 0.3, centre, 1.0e9)
    assert d.shape == (240, 240, 240) and d.dtype == np.float32


# ---- calibration: purity, completeness, adopted cut --------------------------------------------------------------------

def test_purity_completeness():
    """Spec N1: P = n_pure / n_cp, C = n_comp / n_anchor (pooled counts, one value per cut)."""
    P, Cc = C.purity_completeness(np.array([1, 2, 3]), np.array([4, 4, 4]), np.array([3, 2, 1]), np.array([6, 6, 6]))
    np.testing.assert_allclose(P, [0.25, 0.5, 0.75], rtol=1e-15)
    np.testing.assert_allclose(Cc, [0.5, 1 / 3, 1 / 6], rtol=1e-15)


def test_adopted_cut_minimises_abs_log_ratio():
    """Spec N1: adopted cut = min |ln(P/C)|.  Cuts 1..4 with P = .2, .4, .55, .7 and C = .8, .6, .5, .4:
    |ln P/C| = 1.39, 0.41, 0.095, 0.56 -> cut 3, P = 0.55, C = 0.5."""
    cut, P, Cc = C.adopted_cut(np.array([1.0, 2.0, 3.0, 4.0]), np.array([20, 40, 55, 70]), np.array([100] * 4),
                               np.array([160, 120, 100, 80]), np.array([200] * 4))
    assert float(cut) == 3.0
    assert float(P) == pytest.approx(0.55, rel=1e-15) and float(Cc) == pytest.approx(0.5, rel=1e-15)


def test_adopted_cut_uses_the_log_ratio_not_the_difference():
    """cut 1: P = .1, C = .15 (|ln| = 0.405, |P - C| = 0.05); cut 2: P = .6, C = .5 (|ln| = 0.182, |P - C| = 0.1).
    min |ln P/C| picks cut 2 (min |P - C| would pick cut 1)."""
    cut, P, Cc = C.adopted_cut(np.array([1.0, 2.0]), np.array([10, 60]), np.array([100, 100]),
                               np.array([15, 50]), np.array([100, 100]))
    assert float(cut) == 2.0 and float(P) == pytest.approx(0.6) and float(Cc) == pytest.approx(0.5)


@pytest.mark.parametrize("tied,first", [((1, 2), 5.0), ((0, 2), 2.0), ((0, 1, 2), 2.0)])
def test_adopted_cut_ties_go_to_the_first_cut(tied, first):
    """Spec N1 (08_resolution_7k.adopted = min(cuts, key=...)): equal |ln P/C| -> the first of the tied cuts in the
    order given.  Cuts (2, 5, 8); tied cuts have P = 0.5, C = 0.4 (identical counts), the others P = 0.9, C = 0.1."""
    cuts = np.array([2.0, 5.0, 8.0])
    n_pure = np.array([50 if j in tied else 90 for j in range(3)])
    n_comp = np.array([40 if j in tied else 10 for j in range(3)])
    cut, P, Cc = C.adopted_cut(cuts, n_pure, np.array([100] * 3), n_comp, np.array([100] * 3))
    assert float(cut) == first and float(P) == 0.5 and float(Cc) == 0.4


# ---- zoom sample numbers ---------------------------------------------------------------------------------------------

def zoom_table():
    """Six regions; Cluster129 is the second row with kappa 5: one region higher (7), one tie (the other 5)."""
    return dict(region=np.array([5, 129, 7, 300, 12, 40]),
                M200c=np.array([1e13, 8e13, 2e13, 3e13, 5e12, 4e13]),
                R200c=np.array([0.5, 1.0, 0.6, 0.7, 0.4, 0.8]),
                **{"kfix_1.0": np.array([3., 5., 7., 5., 2., 4.]), "kfix_1.5": np.array([3., 6., 5., 4., 2., 5.]),
                   "kfix_2.0": np.array([2., 6., 4., 5., 3., 4.]), "kfix_3.0": np.array([4., 8., 5., 6., 3., 7.]),
                   "kr200_1.5": np.array([2., 6., 5., 4., 1., 3.])})


def test_zoom_numbers_hand_table():
    """Spec N2, N3 on a 6-region table.  Hand: n = 6; kappa129 = 5; n_higher = #(k > 5) = 1; n_at_least = #(k >= 5)
    = 3 (incl. 129); n_tie = 3 - 1 - 1 = 1; kappa(1.5 R200c) of 129 = 6; median_1p5R200 = 1.5 * median(R200c) =
    1.5 * 0.65 = 0.975.  Spearman rho and log10 p of kappa(1 Mpc) vs log10 M200c (scipy.stats.spearmanr); over the
    apertures 1, 1.5, 2, 3: min and max rho, max p."""
    from scipy.stats import spearmanr
    t = zoom_table()
    z = C.zoom_numbers(t)
    lm = np.log10(t["M200c"])
    assert z["n"] == 6
    assert z["k129"] == 5 and z["k129_r200"] == 6
    assert (z["n_higher"], z["n_at_least"], z["n_tie"]) == (1, 3, 1)
    assert z["median_1p5R200"] == pytest.approx(0.975, rel=1e-12)
    r, p = spearmanr(lm, t["kfix_1.0"])
    assert z["rho"] == pytest.approx(r, rel=1e-12) and z["logp"] == pytest.approx(np.log10(p), rel=1e-12)
    rr, pp = zip(*(spearmanr(lm, t[f"kfix_{a}"]) for a in ("1.0", "1.5", "2.0", "3.0")))
    assert z["ap_rho_min"] == pytest.approx(min(rr), rel=1e-12)
    assert z["ap_rho_max"] == pytest.approx(max(rr), rel=1e-12)
    assert z["ap_p_max"] == pytest.approx(max(pp), rel=1e-12)


def test_zoom_numbers_other_region():
    """region=7 (kappa 7, the highest): n_higher 0, n_at_least 1, n_tie 0, kappa(1.5 R200c) = 5."""
    z = C.zoom_numbers(zoom_table(), region=7)
    assert (z["k129"], z["k129_r200"], z["n_higher"], z["n_at_least"], z["n_tie"]) == (7, 5, 0, 1, 0)


def test_zoom_numbers_do_not_depend_on_row_order():
    """Metamorphic: permuting the rows of the table changes no number."""
    t = zoom_table()
    o = np.array([3, 0, 5, 1, 4, 2])
    a, b = C.zoom_numbers(t), C.zoom_numbers({k: v[o] for k, v in t.items()})
    for k in a:
        assert b[k] == pytest.approx(a[k], rel=1e-12), k


# ---- parent-volume numbers -------------------------------------------------------------------------------------------

def parent_tables():
    """4 top haloes (the most massive = Cluster129, kappa 9), 3 control haloes, 3 The300 progenitors (Cluster129
    again, kappa 9; region 7; region 300 which has no zoom); 3k zooms of regions 129, 7, 12."""
    parent = dict(id=np.arange(10),
                  kind=np.array(["top"] * 4 + ["control"] * 3 + ["the300"] * 3),
                  region=np.array([-1] * 7 + [129, 7, 300]),
                  M200c=np.array([1e14, 5e13, 3e13, 2e13, 1e12, 3e12, 1e13, 1e14, 2e13, 3e13]),
                  **{"kfix_1.0": np.array([9., 9., 4., 3., 1., 2., 2., 9., 3., 5.])})
    zooms3k = dict(region=np.array([129, 7, 12]), **{"kfix_1.0": np.array([8., 5., 1.])})
    return parent, zooms3k


def test_parent_numbers_hand_tables():
    """Spec N4, N5.  Fit sample = top + control without the most massive top halo = rows 1..6: n_fit 6, n_top 4,
    n_rand = 6 - 4 + 1 = 3.  OLS kappa = A log10 M + B (numpy polyfit) on the fit sample; rho = Spearman of the fit
    sample; scatter = std of the fit residuals; excess = residual of Cluster129 (row 0), excess_sigma = excess /
    scatter; kappa129_parent = 9; other top haloes with kappa >= 9: 1 (row 1).  N5: The300 haloes with a 3k zoom =
    regions 129, 7: n_cal 2, mean |kappa_parent - kappa_zoom| = (|9 - 8| + |3 - 5|) / 2 = 1.5; zoom kappa of 129 = 8."""
    from scipy.stats import spearmanr
    parent, z3 = parent_tables()
    r = C.parent_numbers(parent, z3)
    lm, k = np.log10(parent["M200c"]), parent["kfix_1.0"]
    fit = np.arange(1, 7)
    A, B = np.polyfit(lm[fit], k[fit], 1)
    res = k - (A * lm + B)
    sig = res[fit].std()
    assert (r["n_top"], r["n_fit"], r["n_rand"]) == (4, 6, 3)
    assert r["slope"] == pytest.approx(A, rel=1e-9) and r["intercept"] == pytest.approx(B, rel=1e-9)
    assert r["rho"] == pytest.approx(spearmanr(lm[fit], k[fit])[0], rel=1e-12)
    assert r["scatter"] == pytest.approx(sig, rel=1e-9)
    assert r["excess"] == pytest.approx(res[0], rel=1e-9)
    assert r["excess_sigma"] == pytest.approx(res[0] / sig, rel=1e-9)
    assert r["k129_parent"] == 9 and r["n_other_ge"] == 1
    assert r["n_cal"] == 2 and r["mean_absdiff"] == pytest.approx(1.5, rel=1e-15) and r["k129_zoom3k"] == 8


def test_parent_numbers_do_not_depend_on_row_order():
    """Metamorphic: permuting the rows of both tables changes no number (Cluster129 is found by mass, not by row)."""
    parent, z3 = parent_tables()
    o = np.array([6, 2, 9, 0, 4, 7, 1, 8, 3, 5])
    a = C.parent_numbers(parent, z3)
    b = C.parent_numbers({k: v[o] for k, v in parent.items()}, {k: v[[2, 0, 1]] for k, v in z3.items()})
    for key in a:
        assert b[key] == pytest.approx(a[key], rel=1e-9), key


# ---- load_data -------------------------------------------------------------------------------------------------------

def test_load_data_mirrors_the_hdf5_groups(tmp_path):
    """load_data returns nested dicts mirroring the groups; a leaf group holds its datasets as numpy arrays and its
    attributes in 'attrs'."""
    import h5py
    f = tmp_path / "t.hdf5"
    with h5py.File(f, "w") as h:
        g = h.create_group("zooms_7k/z4")
        g["region"] = np.array([3, 129])
        g["kfix_1.0"] = np.array([2.0, 9.0])
        g.attrs["redshift"] = 4.038
        g.attrs["snapshot"] = 55
        p = h.create_group("mdpl2_parent")
        p["kind"] = np.array(["top", "control"], dtype=h5py.string_dtype())
        p.attrs["redshift"] = 4.038
    d = C.load_data(f)
    leaf = d["zooms_7k"]["z4"]
    assert isinstance(leaf["region"], np.ndarray)
    np.testing.assert_array_equal(leaf["region"], [3, 129])
    np.testing.assert_array_equal(leaf["kfix_1.0"], [2.0, 9.0])
    assert leaf["attrs"]["redshift"] == pytest.approx(4.038) and int(leaf["attrs"]["snapshot"]) == 55
    kind = [k.decode() if isinstance(k, bytes) else str(k) for k in d["mdpl2_parent"]["kind"]]
    assert kind == ["top", "control"]


# ---- merging ---------------------------------------------------------------------------------------------------------

def test_merge_joins_two_collinear_arcs_at_a_saddle():
    """Arcs (0,0,0)->(1,0,0) and (1,0,0)->(2,0,0), 6 points each, joined at a saddle: one filament of 11 points (the
    junction point appears once), length 2, members [0, 1]."""
    A, B = segment([0, 0, 0], [1, 0, 0], 6), segment([1, 0, 0], [2, 0, 0], 6)
    fils, members = C.merge_arcs([A, B], mergeable([[3, 2], [2, 3]]))
    assert len(fils) == 1 and sorted(members[0]) == [0, 1]
    assert len(fils[0]) == 11 and length(fils[0]) == pytest.approx(2.0, rel=1e-12)
    ends = sorted(map(tuple, np.asarray(fils[0])[[0, -1]].round(12)))
    assert ends == [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]


def test_merge_joins_a_reversed_arc():
    """The second arc listed from (2,0,0) to (1,0,0): still one filament from (0,0,0) to (2,0,0)."""
    A, B = segment([0, 0, 0], [1, 0, 0], 6), segment([2, 0, 0], [1, 0, 0], 6)
    fils, members = C.merge_arcs([A, B], mergeable([[3, 2], [3, 2]]))
    assert len(fils) == 1 and len(fils[0]) == 11
    assert sorted(map(tuple, np.asarray(fils[0])[[0, -1]].round(12))) == [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]


def test_merge_does_not_join_at_a_maximum():
    """Spec P4 (joins only at junctions with a saddle endpoint): the same two arcs meeting at a maximum (type 3) stay
    two filaments."""
    A, B = segment([0, 0, 0], [1, 0, 0], 6), segment([1, 0, 0], [2, 0, 0], 6)
    fils, members = C.merge_arcs([A, B], mergeable([[3, 3], [3, 3]]))
    assert len(fils) == 2 and sorted(sorted(m) for m in members) == [[0], [1]]


@pytest.mark.parametrize("theta,n", [(45.0, 1), (75.0, 2)])
def test_merge_angle_limit(theta, n):
    """Spec P4 (<= 60 deg): the second arc leaves the junction at theta from the straight continuation of the first;
    45 deg -> joined (1 filament), 75 deg -> not joined (2)."""
    t = np.radians(theta)
    A, B = segment([0, 0, 0], [1, 0, 0], 6), segment([1, 0, 0], [1 + np.cos(t), np.sin(t), 0], 6)
    fils, _ = C.merge_arcs([A, B], mergeable([[3, 2], [2, 3]]))
    assert len(fils) == n


@pytest.mark.parametrize("gap,n", [(0.3, 1), (0.8, 2)])
def test_merge_endpoint_tolerance(gap, n):
    """Spec P4 (0.5 h^-1 Mpc tolerance): collinear arcs whose facing ends are 0.3 apart join, 0.8 apart do not."""
    A, B = segment([0, 0, 0], [1, 0, 0], 6), segment([1 + gap, 0, 0], [2 + gap, 0, 0], 6)
    fils, _ = C.merge_arcs([A, B], mergeable([[3, 2], [2, 3]]))
    assert len(fils) == n


def test_merge_y_junction_keeps_the_straight_pair():
    """Three arcs meet at a saddle at the origin: from (-1,0,0), to (1,0,0) and to (0,1,0).  The straight pair joins,
    the perpendicular arc stays alone: filaments {0, 1} and {2}; each arc in exactly one filament."""
    arcs = [segment([-1, 0, 0], [0, 0, 0], 5), segment([0, 0, 0], [1, 0, 0], 5), segment([0, 0, 0], [0, 1, 0], 5)]
    fils, members = C.merge_arcs(arcs, mergeable([[3, 2], [2, 3], [2, 3]]))
    assert sorted(sorted(int(i) for i in m) for m in members) == [[0, 1], [2]]


def test_merge_open_boundaries():
    """Spec P4 (open boundaries): arcs ending at x = +14.9 and starting at x = -14.9 (0.2 apart through a periodic
    30 h^-1 Mpc box, 29.8 apart in the open cube) are not joined."""
    A, B = segment([10, 0, 0], [14.9, 0, 0], 6), segment([-14.9, 0, 0], [-10, 0, 0], 6)
    fils, _ = C.merge_arcs([A, B], mergeable([[3, 2], [2, 3]]))
    assert len(fils) == 2


def random_network(rng):
    """A random tree of arcs: nodes in the cube at least 1 h^-1 Mpc apart (so no two junctions fall within the 0.5
    tolerance), node i > 0 linked to a random earlier node, arc direction random; arc ends sit exactly on their
    nodes; interior points jittered by 0.1; node types 2 (saddle), 3 (maximum), 4 (bifurcation) at random.  A tree
    has no loops, so every join removes exactly one duplicated junction point."""
    n_nodes = int(rng.integers(3, 12))
    nodes = []
    while len(nodes) < n_nodes:
        p = rng.uniform(-6, 6, 3)
        if all(np.linalg.norm(p - q) >= 1.0 for q in nodes):
            nodes.append(p)
    ntype = rng.choice([2, 3, 4], n_nodes)
    arcs, types = [], []
    for i in range(1, n_nodes):
        j = int(rng.integers(0, i))
        a_, b_ = (i, j) if rng.random() < 0.5 else (j, i)
        m = int(rng.integers(2, 12))
        a = segment(nodes[a_], nodes[b_], m)
        a[1:-1] += 0.1 * rng.standard_normal((m - 2, 3))
        arcs.append(a)
        types.append([ntype[a_], ntype[b_]])
    return arcs, np.array(types, int).reshape(-1, 2)


@pytest.mark.parametrize("seed", range(40))
def test_merge_uses_each_arc_once_and_conserves_length(seed):
    """Spec P4: every input arc in exactly one filament.  Because the arcs share their junction points exactly and
    a joined junction point appears once, the total length is conserved and the number of points drops by one per
    join: sum len(f) = sum len(a) - (N_arcs - N_filaments)."""
    arcs, types = random_network(np.random.default_rng(seed))
    fils, members = C.merge_arcs(arcs, mergeable(types))
    assert sorted(int(i) for m in members for i in m) == list(range(len(arcs)))
    assert len(fils) == len(members)
    assert sum(length(f) for f in fils) == pytest.approx(sum(length(a) for a in arcs), rel=1e-9)
    assert sum(len(f) for f in fils) == sum(len(a) for a in arcs) - (len(arcs) - len(fils))


# ---- smoothing -------------------------------------------------------------------------------------------------------

def test_smooth_keeps_a_straight_evenly_spaced_line():
    """Neighbour averaging maps evenly spaced collinear points onto themselves."""
    a = segment([-3.0, 1.0, 2.0], [4.0, -2.0, 5.5], 15)
    np.testing.assert_allclose(C.smooth_filament(a), a, atol=1e-9)


def test_smooth_one_iteration_three_points():
    """(0,0,0), (1,1,0), (2,0,0), one iteration: the middle point becomes the mean of its neighbours (1,0,0); the
    endpoints stay."""
    a = np.array([[0.0, 0, 0], [1, 1, 0], [2, 0, 0]])
    np.testing.assert_allclose(C.smooth_filament(a, niter=1), [[0, 0, 0], [1, 0, 0], [2, 0, 0]], atol=1e-12)


def test_smooth_updates_all_points_from_the_previous_iteration():
    """(0,0,0), (1,1,0), (2,1,0), (3,0,0), one iteration, every interior point from the old positions:
    p1 = (p0 + p2)/2 = (1, 0.5, 0), p2 = (p1 + p3)/2 = (2, 0.5, 0).  (Updating in place would give p2 = (2, 0.25, 0).)"""
    a = np.array([[0.0, 0, 0], [1, 1, 0], [2, 1, 0], [3, 0, 0]])
    np.testing.assert_allclose(C.smooth_filament(a, niter=1),
                               [[0, 0, 0], [1, 0.5, 0], [2, 0.5, 0], [3, 0, 0]], atol=1e-12)


def test_smooth_two_points_unchanged():
    a = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 2.0]])
    np.testing.assert_allclose(C.smooth_filament(a), a, atol=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_smooth_fixes_endpoints_and_keeps_the_number_of_points(seed):
    """Spec P5: endpoints fixed; same shape as the input."""
    rng = np.random.default_rng(seed)
    a = np.cumsum(rng.standard_normal((int(rng.integers(3, 40)), 3)), axis=0) - 2.0
    s = C.smooth_filament(a)
    assert s.shape == a.shape
    np.testing.assert_allclose(s[[0, -1]], a[[0, -1]], atol=1e-9)


# ---- postprocess -----------------------------------------------------------------------------------------------------

def test_postprocess_drops_arcs_shorter_than_0_01():
    """Spec P5: arcs with length < 0.01 are dropped: a 0.0099 arc goes, a 0.0101 arc and a long arc stay."""
    arcs = [segment([0, 0, 0], [1, 0, 0], 5), segment([3, 3, 3], [3.0099, 3, 3], 2),
            segment([-3, 2, 1], [-3, 2, 1.0101], 2)]
    out = C.postprocess(arcs, np.array([[3, 3], [3, 3], [3, 3]]))
    assert len(out) == 2
    assert sorted(round(length(f), 6) for f in out) == [0.0101, 1.0]


def test_postprocess_drops_single_point_arcs():
    """Spec P5: arcs with fewer than 2 points are dropped."""
    out = C.postprocess([segment([0, 0, 0], [1, 0, 0], 5), np.array([[2.0, 2.0, 2.0]])], np.array([[3, 3], [3, 3]]))
    assert len(out) == 1


def test_postprocess_keeps_end_types_aligned_after_dropping():
    """A tiny arc (maximum ends) listed first, then two collinear arcs meeting at a saddle: after the tiny arc is
    dropped the remaining arcs must keep their own end types, so they are joined: 1 filament of length 2 from
    (0,0,0) to (2,0,0).  (End types shifted by one would put a maximum at the junction: 2 filaments.)"""
    arcs = [segment([5, 5, 5], [5, 5, 5.001], 2), segment([0, 0, 0], [1, 0, 0], 6), segment([1, 0, 0], [2, 0, 0], 6)]
    out = C.postprocess(arcs, np.array([[3, 3], [3, 2], [2, 3]]))
    assert len(out) == 1
    assert length(out[0]) == pytest.approx(2.0, rel=1e-9)
    assert sorted(map(tuple, np.asarray(out[0])[[0, -1]].round(9))) == [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]


@pytest.mark.parametrize("seed", range(20))
def test_postprocess_keeps_endpoints_and_uses_each_arc_once(seed):
    """Spec P5: merge (each arc once) then smooth with endpoints fixed.  Hence the number of output points is the
    input number minus one per join, and every output endpoint is an input arc endpoint."""
    arcs, types = random_network(np.random.default_rng(100 + seed))
    out = C.postprocess(arcs, types)
    n_join = len(arcs) - len(out)
    assert n_join >= 0
    assert sum(len(f) for f in out) == sum(len(a) for a in arcs) - n_join
    ends = np.array([e for a in arcs for e in (a[0], a[-1])])
    for f in out:
        for e in (f[0], f[-1]):
            assert np.linalg.norm(ends - e, axis=1).min() < 1e-9


# ---- kappa -----------------------------------------------------------------------------------------------------------

def test_kappa_hand_counts():
    """Spec P6, radius 1 about the origin: a radial line (0,0,0)->(2,0,0) crosses (1); a line through the sphere from
    (-2,0.3,0) to (2,0.3,0) crosses twice but counts once (1); a line inside (0.2..0.8) and one outside (2..3) do not
    count: kappa = 2."""
    fils = [segment([0, 0, 0], [2, 0, 0]), segment([-2, 0.3, 0], [2, 0.3, 0], 41),
            segment([0, 0.2, 0], [0, 0.8, 0]), segment([0, 0, 2], [0, 0, 3])]
    assert C.count_kappa(fils, 1.0) == 2


def test_kappa_boundary_point_is_inside():
    """Spec P6 (<= radius is inside): points (0,1,0), (0,2,0), (0,3,0) with radius 1: the first is inside, so the
    filament counts; a filament whose points all lie at distance exactly 1 is entirely inside and does not count."""
    on_sphere = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0]])
    assert C.count_kappa([np.array([[0, 1.0, 0], [0, 2.0, 0], [0, 3.0, 0]])], 1.0) == 1
    assert C.count_kappa([on_sphere], 1.0) == 0


def test_kappa_counts_points_not_segments():
    """Spec P6 (at least one POINT inside): a two-point filament from (-2,0,0) to (2,0,0) passes through the sphere
    but has no sampling point inside: kappa = 0."""
    assert C.count_kappa([np.array([[-2.0, 0, 0], [2.0, 0, 0]])], 1.0) == 0


def test_kappa_centre_argument():
    """Shifting the filaments by c and passing centre = c gives the same count (3 radial lines out of c, one not
    reaching the shell)."""
    c = np.array([3.0, -2.0, 7.5])
    fils = [segment([0, 0, 0], [2, 0, 0]), segment([0, 0, 0], [0, -1.5, 0]), segment([0, 0, 0], [0, 0, 0.5])]
    assert C.count_kappa(fils, 1.0) == 2
    assert C.count_kappa([f + c for f in fils], 1.0, centre=c) == 2
    assert C.count_kappa([f + c for f in fils], 1.0) == 0


def test_kappa_empty():
    assert C.count_kappa([], 1.0) == 0


@pytest.mark.parametrize("seed", range(10))
def test_kappa_translation_invariant(seed):
    """Metamorphic: translating filaments and centre together does not change kappa."""
    rng = np.random.default_rng(seed)
    fils = [np.cumsum(0.3 * rng.standard_normal((int(rng.integers(2, 30)), 3)), 0) + rng.uniform(-2, 2, 3)
            for _ in range(int(rng.integers(1, 20)))]
    c, s = rng.uniform(-1, 1, 3), rng.uniform(-50, 50, 3)
    r = float(rng.uniform(0.3, 2.0))
    assert C.count_kappa(fils, r, centre=c) == C.count_kappa([f + s for f in fils], r, centre=c + s)


# ---- parse_skeleton --------------------------------------------------------------------------------------------------

def test_parse_axis_order_and_mapping(tmp_path):
    """Spec P3: file pixel (k, j, i) -> (x, y, z) = -15 + (i, j, k) * 0.125.  Saddle at file (10, 20, 30) ->
    (-11.25, -12.5, -13.75); maximum at file (12, 26, 41) -> (-9.875, -11.75, -13.5).  End types (2, 3)."""
    P0, P1 = np.array([10.0, 20, 30]), np.array([12.0, 26, 41])
    pts = np.array([P0, [11.0, 23, 35], P1])
    f = write_ndskl(tmp_path / "s.NDskl", [(2, P0, 0), (3, P1, 0)], [(0, 1, pts)])
    arcs, et, mx = C.parse_skeleton(f)
    assert len(arcs) == 1
    np.testing.assert_allclose(arcs[0], [[-11.25, -12.5, -13.75], [-10.625, -12.125, -13.625],
                                         [-9.875, -11.75, -13.5]], atol=1e-9)
    np.testing.assert_array_equal(np.asarray(et).reshape(-1, 2), [[2, 3]])
    np.testing.assert_allclose(np.asarray(mx).reshape(-1, 3), [[-9.875, -11.75, -13.5]], atol=1e-9)


def test_parse_uses_lhalf_and_cell(tmp_path):
    """Same file read with lhalf 4, cell 0.25: (x, y, z) = -4 + (i, j, k) * 0.25."""
    P0, P1 = np.array([1.0, 2, 3]), np.array([5.0, 6, 9])
    f = write_ndskl(tmp_path / "s.NDskl", [(2, P0, 0), (3, P1, 0)], [(0, 1, np.array([P0, P1]))], bbox=32)
    arcs, et, mx = C.parse_skeleton(f, lhalf=4.0, cell=0.25)
    np.testing.assert_allclose(arcs[0], to_xyz([P0, P1], 4.0, 0.25), atol=1e-9)
    np.testing.assert_allclose(np.asarray(mx).reshape(-1, 3), to_xyz([P1], 4.0, 0.25), atol=1e-9)


def boundary_file(path):
    """Four arcs; CP 2 (saddle) has boundary flag 1, CP 3 (maximum) flag 2 (any non-zero = on the boundary).
        A0: cp1 (saddle) -> cp0 (max)          kept,    end types (2, 3)
        A1: cp2 (saddle, flagged) -> cp4 (4)   dropped
        A2: cp1 (saddle) -> cp3 (max, flagged) dropped
        A3: cp0 (max) -> cp5 (saddle)          kept,    end types (3, 2)
    """
    P = {0: (100, 100, 100), 1: (100, 100, 110), 2: (100, 100, 0), 3: (0, 100, 110), 4: (100, 100, 40),
         5: (100, 60, 100)}
    cps = [(3, P[0], 0), (2, P[1], 0), (2, P[2], 1), (3, P[3], 2), (4, P[4], 0), (2, P[5], 0)]
    arcs = [(1, 0, segment(P[1], P[0], 6)), (2, 4, segment(P[2], P[4], 11)), (1, 3, segment(P[1], P[3], 21)),
            (0, 5, segment(P[0], P[5], 9))]
    return write_ndskl(path, cps, arcs), P


def test_parse_drops_boundary_arcs_and_returns_types_and_maxima(tmp_path):
    """Spec P3: arcs ending at a boundary-flagged critical point are dropped (A1, A2); end types of the kept arcs
    are (2, 3) and (3, 2); maxima = all critical points of type 3 (CP 0 and CP 3)."""
    f, P = boundary_file(tmp_path / "b.NDskl")
    arcs, et, mx = C.parse_skeleton(f)
    assert len(arcs) == 2
    np.testing.assert_allclose(arcs[0], to_xyz(segment(P[1], P[0], 6)), atol=1e-9)
    np.testing.assert_allclose(arcs[1], to_xyz(segment(P[0], P[5], 9)), atol=1e-9)
    np.testing.assert_array_equal(np.asarray(et).reshape(-1, 2), [[2, 3], [3, 2]])
    mx = np.asarray(mx).reshape(-1, 3)
    want = to_xyz([P[0], P[3]])
    np.testing.assert_allclose(mx[np.lexsort(mx.T)], want[np.lexsort(want.T)], atol=1e-9)


def test_parse_no_maxima_gives_empty_array(tmp_path):
    """Maxima come back as an (0, 3) array when the skeleton has no type-3 point."""
    P0, P1 = np.array([10.0, 10, 10]), np.array([10.0, 10, 20])
    f = write_ndskl(tmp_path / "s.NDskl", [(2, P0, 0), (4, P1, 0)], [(0, 1, np.array([P0, P1]))])
    _, et, mx = C.parse_skeleton(f)
    assert np.asarray(mx).reshape(-1, 3).shape == (0, 3)
    np.testing.assert_array_equal(np.asarray(et).reshape(-1, 2), [[2, 4]])


def wrapped_file(path, axis, jump, bbox=240):
    """One arc of 4 points along file column `axis`: a+2, a, a+jump, a+jump-2 with a = (bbox - jump)/2, so the
    middle step is `jump` pixels; for jump 234 of 240 the arc leaves through the low face and comes back through the
    high face (a periodic run)."""
    a = (bbox - jump) / 2.0
    pts = np.full((4, 3), bbox / 2.0)
    pts[:, axis] = [a + 2.0, a, a + jump, a + jump - 2.0]
    return write_ndskl(path, [(2, pts[0], 0), (3, pts[-1], 0)], [(0, 1, pts)], bbox=bbox)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_parse_raises_on_a_wrapped_arc(tmp_path, axis):
    """Spec P3: an arc stepping more than half the cube (234 of 240 pixels) means DisPerSE ran periodic: raise."""
    f = wrapped_file(tmp_path / "w.NDskl", axis, 234)
    raises_error(lambda: C.parse_skeleton(f))


def test_parse_accepts_steps_below_half_the_cube(tmp_path):
    """A 100-pixel step (< 120) is not a wrap: no error, one arc."""
    arcs, _, _ = C.parse_skeleton(wrapped_file(tmp_path / "w.NDskl", 0, 100))
    assert len(arcs) == 1


# ---- run_disperse (with stand-in binaries that record their calls) ------------------------------------------------------

FAKE_MSE = r'''#!{py}
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
mode = (HERE / "MODE").read_text().strip()
args = sys.argv[1:]
with open(HERE / "calls.jsonl", "a") as f:
    f.write(json.dumps(dict(prog="mse", args=args, omp=os.environ.get("OMP_NUM_THREADS"))) + "\n")
src = Path(args[0]) if Path(args[0]).is_absolute() else Path.cwd() / args[0]
if not src.exists():
    print("ERROR: no input file", src); sys.exit(1)
cut = args[args.index("-cut") + 1]
if "-loadMSC" in args:
    print("Enforcing boundary conditions: natural -> natural. SKIPPING.")
    if mode == "skip_load":
        print("Skipping mask")
else:
    if mode != "nomask":
        print("Building mask ... done. (6348 2-faces on boundary)")
    if mode == "disabled":
        print("Boundary detection disabled")
    Path(src.name + ".MSC").write_text("stand-in MSC")
Path(f"{{src.name}}_c{{cut}}.up.NDskl").write_text(f"cut={{cut}}\n")
'''

FAKE_SKELCONV = r'''#!{py}
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
args = sys.argv[1:]
with open(HERE / "calls.jsonl", "a") as f:
    f.write(json.dumps(dict(prog="skelconv", args=args, omp=os.environ.get("OMP_NUM_THREADS"))) + "\n")
src = Path(args[0]) if Path(args[0]).is_absolute() else Path.cwd() / args[0]
Path(src.name + ".BRK.S008.a.NDskl").write_text(src.read_text())
'''


@pytest.fixture
def fake_bin(tmp_path):
    b = tmp_path / "bin"
    b.mkdir()
    for name, txt in (("mse", FAKE_MSE), ("skelconv", FAKE_SKELCONV)):
        p = b / name
        p.write_text(txt.format(py=sys.executable))
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (b / "MODE").write_text("ok")
    work = tmp_path / "work"
    work.mkdir()
    return b, work


def calls(b):
    return [json.loads(x) for x in (b / "calls.jsonl").read_text().splitlines()]


def opt(args, name):
    return args[args.index(name) + 1]


def test_run_disperse_commands(fake_bin):
    """Spec P7: `mse d.fits -periodicity 0 -cut 12 -upSkl -forceLoops`, then `mse d.fits -loadMSC d.fits.MSC
    -cut <cut> -upSkl`, then `skelconv <skeleton of that cut> -breakdown -smooth 8 -to NDskl_ascii`, each with
    OMP_NUM_THREADS=1; the returned file is the converted skeleton of cut 3 (not of the fresh run's cut 12)."""
    b, work = fake_bin
    delta = np.random.default_rng(0).standard_normal((8, 8, 8)).astype(np.float32)
    out = C.run_disperse(delta, work, 3, str(b))
    c = calls(b)
    assert [x["prog"] for x in c] == ["mse", "mse", "skelconv"]
    assert all(x["omp"] == "1" for x in c)
    fresh, load, conv = (x["args"] for x in c)
    assert Path(fresh[0]).name == "d.fits"
    assert opt(fresh, "-periodicity") == "0" and float(opt(fresh, "-cut")) == 12.0
    assert "-upSkl" in fresh and "-forceLoops" in fresh and "-loadMSC" not in fresh and "-nsig" not in fresh
    assert Path(load[0]).name == "d.fits" and Path(opt(load, "-loadMSC")).name == "d.fits.MSC"
    assert float(opt(load, "-cut")) == 3.0 and "-upSkl" in load
    assert opt(conv, "-smooth") == "8" and opt(conv, "-to") == "NDskl_ascii" and "-breakdown" in conv
    assert Path(out).exists()
    assert float(Path(out).read_text().strip().split("=")[1]) == 3.0


def test_run_disperse_fresh_cut(fake_bin):
    """fresh_cut sets the cut of the fresh (MSC-building) run."""
    b, work = fake_bin
    C.run_disperse(np.zeros((8, 8, 8), np.float32) + 0.5, work, 5, str(b), fresh_cut=8.0)
    fresh = calls(b)[0]["args"]
    assert float(opt(fresh, "-cut")) == 8.0 and "-loadMSC" not in fresh


@pytest.mark.parametrize("mode", ["nomask", "disabled", "skip_load"])
def test_run_disperse_raises_without_the_boundary_mask(fake_bin, mode):
    """Spec P7: raise if the fresh mse run does not report "Building mask" (nomask), or any run reports that it
    disabled or skipped the mask ("Boundary detection disabled", "Skipping mask").  The ordinary loadMSC line
    "Enforcing boundary conditions: natural -> natural. SKIPPING." is printed in every mode and must not raise
    (test_run_disperse_commands)."""
    b, work = fake_bin
    (b / "MODE").write_text(mode)
    raises_error(lambda: C.run_disperse(np.ones((8, 8, 8), np.float32), work, 3, str(b)))


# ---- end to end with DisPerSE on a synthetic field (slow) -------------------------------------------------------------

def disperse_bin():
    d = os.environ.get("DISPERSE_BIN") or (os.path.dirname(shutil.which("mse")) if shutil.which("mse") else None)
    if not d or not (Path(d) / "mse").exists() or not (Path(d) / "skelconv").exists():
        pytest.skip("DisPerSE not found: set DISPERSE_BIN to the folder holding mse and skelconv")
    return d


CA, CB = np.array([1.0, -1.5, 0.5]), np.array([-1.0, 1.25, -0.75])     # clump centres on grid nodes


@pytest.fixture(scope="module")
def synthetic_skeleton(tmp_path_factory):
    """Two dense clumps at known node positions (asymmetric in x, y, z) joined by a thin ridge, on a weak uniform
    background, in a cube of +-4 h^-1 Mpc (64^3 cells of 0.125) about the origin."""
    b = disperse_bin()
    rng = np.random.default_rng(42)
    clumps = [c + 0.02 * rng.standard_normal((20000, 3)) for c in (CA, CB)]
    t = rng.uniform(0, 1, 3000)[:, None]
    ridge = CA + t * (CB - CA) + 0.03 * rng.standard_normal((3000, 3))
    bg = rng.uniform(-4.0, 4.0 - 1e-6, (30000, 3))
    delta = C.density_field(np.vstack(clumps + [ridge, bg]), np.zeros(3), 1.0e9, lhalf=4.0, cell=0.125, rs=0.5)
    work = tmp_path_factory.mktemp("disperse")
    path = C.run_disperse(delta, work, 0.2 * float(delta.max()), b)
    return C.parse_skeleton(path, lhalf=4.0, cell=0.125)


@pytest.mark.slow
def test_disperse_maxima_at_the_clumps(synthetic_skeleton):
    """Exactly two maxima, each within 0.03 h^-1 Mpc of a clump centre (an axis swap or a half-cell offset would put
    them elsewhere)."""
    _, _, mx = synthetic_skeleton
    mx = np.asarray(mx).reshape(-1, 3)
    assert len(mx) == 2, mx.tolist()
    for c in (CA, CB):
        assert np.linalg.norm(mx - c, axis=1).min() < 0.03


@pytest.mark.slow
def test_disperse_ridge_is_one_filament(synthetic_skeleton):
    """The ridge is one filament: kappa(1 h^-1 Mpc) = 1 about each clump (filaments ending on the cube faces are
    dropped by the boundary flag)."""
    arcs, et, _ = synthetic_skeleton
    fils = C.postprocess(arcs, et)
    for c in (CA, CB):
        assert C.count_kappa(fils, 1.0, centre=c) == 1


# ---- the numbers quoted in the paper (spec N6) -------------------------------------------------------------------------

QUOTED = [  # key, value in the text, decimals (None = integer, exact)
    ("n_hosts_z4", 7580, None), ("n_hosts_z5", 2349, None), ("cut_z4", 3, None), ("cut_z5", 5, None),
    ("P_z4", 0.48, 2), ("C_z4", 0.52, 2), ("P_z5", 0.50, 2), ("C_z5", 0.50, 2),
    ("median_1p5R200_z4", 0.5, 1), ("median_1p5R200_z5", 0.35, 2), ("n_z4", 321, None), ("rho_z4", 0.37, 2),
    ("logp_z4", -11, 0), ("ap_rho_min", 0.19, 2), ("ap_rho_max", 0.37, 2), ("k129_z4", 9, None),
    ("k129_z5", 4, None), ("n_higher_z5", 7, None), ("n_tie_z5", 49, None), ("pk_n_top", 186, None),
    ("pk_n_rand", 300, None), ("pk_cal_n", 316, None), ("pk_cal_absdiff", 0.17, 2), ("k129_zoom3k", 8, None),
    ("pk_k129_parent", 7, None), ("pk_n_fit", 485, None), ("pk_slope", 1.7, 1), ("pk_icpt", -18.3, 1),
    ("pk_rho", 0.55, 2), ("pk_scatter", 1.1, 1), ("pk_excess", 1.5, 1), ("pk_excess_sig", 1.4, 1),
    ("pk_n_other_ge", 5, None),
]


@pytest.fixture(scope="module")
def numbers():
    if not DATA.exists():
        pytest.skip(f"{DATA} not present")
    return C.paper_numbers(C.load_data(DATA))


@pytest.mark.parametrize("key,quoted,nd", QUOTED, ids=[q[0] for q in QUOTED])
def test_paper_numbers_round_to_the_text(numbers, key, quoted, nd):
    """Spec N6: each number rounds to the value printed in the paper (p = 1e-11 is checked as round(log10 p) =
    -11)."""
    v = numbers[key]
    if nd is None:
        assert float(v) == float(quoted) and float(v) == int(v)
    else:
        assert round(float(v), nd) == pytest.approx(quoted, abs=1e-12)


def test_paper_numbers_cluster129_at_1p5_r200(numbers):
    """Spec N3: kappa of Cluster129 at 1.5 R200c = 9 (z = 4.038) and 4 (z = 5.017); at z = 4.038 no region has a
    higher kappa and it is the only one with kappa >= 9."""
    assert numbers["k129_r200_z4"] == 9 and numbers["k129_r200_z5"] == 4
    assert numbers["n_at_least_z4"] == 1
