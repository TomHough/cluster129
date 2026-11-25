"""
Cosmic-web connectivity of the The Three Hundred cluster progenitors at z = 4.038 and 5.017 (Methods, "Connectivity").

Used by Connectivity.ipynb.  Three parts:

1. Numbers.  `load_data` reads data/connectivity_cluster129.hdf5 (the processed connectivity of all regions, the
   calibration counts, the MDPL2 parent-volume comparison and the Cluster129 skeletons); `paper_numbers` recomputes
   every number quoted in the Methods from it.
2. Figure.  `plot_web` draws the Extended Data figure (cosmic web around Cluster129) from the same file;
   `slab_image` recomputes its background from the GIZMO-SIMBA 7k snapshot.
3. Pipeline for one region.  `measure_region` runs the method on a zoom: high-resolution dark matter in a
   (30 Mpc/h)^3 cube -> cloud-in-cell overdensity on a 240^3 grid -> Gaussian smoothing (R_s = 0.5 Mpc/h) ->
   DisPerSE (absolute persistence cut, open boundaries) -> post-processing (Singh et al. 2026: drop boundary and
   very short arcs, join arcs that continue each other through a saddle point, smooth) -> connectivity kappa, the
   number of filaments that cross a sphere around the progenitor.  It needs DisPerSE v0.9.25 (mse, skelconv) and
   the simulation data on /data20.

Units: comoving Mpc/h and Msun/h, as in the simulation catalogues.
"""
import glob
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy import ndimage, stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------------------------------- constants
LHALF = 15.0                 # half side of the cube [Mpc/h]
CELL = 0.125                 # grid cell [Mpc/h] -> 240^3 grid
RS = 0.5                     # Gaussian smoothing scale [Mpc/h]
OMEGA_M, OMEGA_B = 0.307115, 0.048           # MDPL2 / The Three Hundred cosmology
RHO_CRIT0 = 2.775e11         # critical density [h^2 Msun Mpc^-3]
RHO_DM_MEAN = (OMEGA_M - OMEGA_B) * RHO_CRIT0  # cosmic mean dark-matter density [h^2 Msun Mpc^-3]
APERTURES = (1.0, 1.5, 2.0, 3.0)             # fixed comoving apertures [Mpc/h]; 1 Mpc/h is the main measurement
FRESH_CUT = 12.0             # cut of the first mse run, which computes the Morse-Smale complex once
MERGE_ANGLE, MERGE_TOL, N_TANGENT, SMOOTH_ITER = 60.0, 0.5, 5, 10   # post-processing (Singh et al. 2026)
_OFFSET, _BOX = 500.0, 1000.0                # frame of the smoothing step (see smooth_filament)



# ======================================================================================================== 1. numbers
def load_data(path="data/connectivity_cluster129.hdf5"):
    """The HDF5 file as nested dicts: data[group][subgroup][dataset] -> numpy array; each group holding datasets also
    has an 'attrs' dict.  Byte strings are decoded to str."""
    import h5py

    def conv(x):
        x = x[()]
        if isinstance(x, np.ndarray) and x.dtype.kind == "S":
            return x.astype(str)
        return x

    def walk(g):
        out = {}
        for k, v in g.items():
            out[k] = walk(v) if isinstance(v, h5py.Group) else conv(v)
        out["attrs"] = {a: (b.decode() if isinstance(b, bytes) else b) for a, b in g.attrs.items()}
        return out
    with h5py.File(path, "r") as h:
        return walk(h)


def purity_completeness(n_pure, n_cp, n_comp, n_anchor):
    """Pooled purity (maxima with a host within that host's R200c / all maxima) and completeness (hosts with a maximum
    within their R200c / all hosts), one value per persistence cut."""
    n_pure, n_cp, n_comp, n_anchor = (np.asarray(x, float) for x in (n_pure, n_cp, n_comp, n_anchor))
    return n_pure / n_cp, n_comp / n_anchor


def adopted_cut(cuts, n_pure, n_cp, n_comp, n_anchor):
    """The cut at which purity and completeness are closest to equal (minimal |ln P/C|; the first such cut on a tie),
    with its purity and completeness."""
    if np.any(np.asarray(n_cp) <= 0) or np.any(np.asarray(n_anchor) <= 0):
        raise ValueError("a cut without density maxima or without hosts: purity or completeness undefined")
    P, C = purity_completeness(n_pure, n_cp, n_comp, n_anchor)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.abs(np.log(P / C))
    if np.isnan(score).any():
        raise ValueError("purity and completeness both zero at a cut: their ratio is undefined")
    j = int(np.argmin(score))                      # the first cut on a tie
    return float(cuts[j]), float(P[j]), float(C[j])


def _index(region, r):
    w = np.where(np.asarray(region).astype(int) == r)[0]
    if len(w) != 1:
        raise ValueError(f"region {r} appears {len(w)} times")
    return int(w[0])


def zoom_numbers(table, region=129):
    """Connectivity of the zoom sample at one epoch (kappa within the fixed 1 Mpc/h aperture unless stated):
    n, Spearman rho and log10 p of kappa against log10 M200c, the region's kappa (also at 1.5 R200c), the numbers of
    progenitors with a higher kappa, with at least its kappa (itself included) and with the same kappa (itself
    excluded), the median 1.5 R200c, and the range of rho and the largest p over the apertures 1, 1.5, 2, 3 Mpc/h."""
    lm = np.log10(np.asarray(table["M200c"], float)); k = np.asarray(table["kfix_1.0"], float)
    i = _index(table["region"], region)
    r, p = stats.spearmanr(lm, k)
    rr, pp = zip(*(stats.spearmanr(lm, np.asarray(table[f"kfix_{a}"], float)) for a in APERTURES))
    n_higher, n_at_least = int((k > k[i]).sum()), int((k >= k[i]).sum())
    return dict(n=len(k), rho=float(r), logp=float(np.log10(max(p, 1e-300))), k129=float(k[i]),
                k129_r200=float(table["kr200_1.5"][i]), n_higher=n_higher, n_at_least=n_at_least,
                n_tie=n_at_least - n_higher - 1, median_1p5R200=float(1.5 * np.median(table["R200c"])),
                ap_rho_min=float(min(rr)), ap_rho_max=float(max(rr)), ap_p_max=float(max(pp)))


def parent_numbers(parent, zooms3k, region=129):
    """Cluster129 among haloes selected by their z=4 mass in the MDPL2 parent volume.
    Fit sample: all 'top' haloes (M200c >= 1e13 Msun/h) and the random 'control' haloes, without Cluster129 (the most
    massive 'top' halo).  Linear least-squares fit kappa = A log10 M200c + B, Spearman rho, scatter = standard
    deviation of the residuals; Cluster129's residual, in filaments and in units of the scatter; the number of other
    'top' haloes with at least its kappa.  Check against the zooms: the 'the300' haloes (the zoom progenitors measured
    in the parent volume) whose region has a 3k-zoom kappa: their number and mean absolute difference, and
    Cluster129's kappa in the 3k zoom and in the parent volume (its 'the300' entry, the halo matched to the zoom;
    it must equal the kappa of the most massive 'top' halo, the same object found by mass)."""
    if region != 129:
        raise ValueError("only Cluster129 is identified in the parent volume by mass (the most massive 'top' halo)")
    kind = np.asarray(parent["kind"]).astype(str); reg = np.asarray(parent["region"]).astype(int)
    lm = np.log10(np.asarray(parent["M200c"], float)); k = np.asarray(parent["kfix_1.0"], float)
    top = np.where(kind == "top")[0]; u = np.where(np.isin(kind, ["top", "control"]))[0]
    i129 = top[np.argmax(np.asarray(parent["M200c"])[top])]
    fit = u[u != i129]; tt = top[top != i129]
    A, B = np.polyfit(lm[fit], k[fit], 1)
    res = k - (A * lm + B); sig = res[fit].std()
    kz = dict(zip(np.asarray(zooms3k["region"]).astype(int), np.asarray(zooms3k["kfix_1.0"], float)))
    t = np.where((kind == "the300") & np.isin(reg, list(kz)))[0]
    kzoom = np.array([kz[r] for r in reg[t]])
    k_match = k[np.where((kind == "the300") & (reg == region))[0]]
    if len(k_match) != 1 or k_match[0] != k[i129]:
        raise ValueError(f"region {region}: parent-volume kappa of its matched entry {k_match} differs from that of "
                         f"the most massive 'top' halo ({k[i129]})")
    return dict(n_top=len(top), n_rand=len(fit) - len(top) + 1, n_fit=len(fit), slope=float(A), intercept=float(B),
                rho=float(stats.spearmanr(lm[fit], k[fit])[0]), scatter=float(sig), k129_parent=float(k_match[0]),
                excess=float(res[i129]), excess_sigma=float(res[i129] / sig), n_other_ge=int((k[tt] >= k[i129]).sum()),
                n_cal=len(t), mean_absdiff=float(np.mean(np.abs(k[t] - kzoom))), k129_zoom3k=float(kz[region]))


def paper_numbers(data):
    """Every number of the Methods "Connectivity" paragraph, recomputed from load_data()."""
    v = {}
    for z in (4, 5):
        c = data["calibration_7k"][f"z{z}"]
        cut, P, C = adopted_cut(c["cuts"], c["n_pure"], c["n_cp"], c["n_comp"], c["n_anchor"])
        v[f"cut_z{z}"], v[f"P_z{z}"], v[f"C_z{z}"], v[f"n_hosts_z{z}"] = cut, P, C, int(c["n_anchor"][0])
        c3 = data["calibration_3k"][f"z{z}"]
        v[f"cut3k_z{z}"] = adopted_cut(c3["cuts"], c3["n_pure"], c3["n_cp"], c3["n_comp"], c3["n_anchor"])[0]
        zn = zoom_numbers(data["zooms_7k"][f"z{z}"])
        v[f"n_z{z}"], v[f"median_1p5R200_z{z}"] = zn["n"], zn["median_1p5R200"]
        v[f"k129_z{z}"], v[f"k129_r200_z{z}"], v[f"n_at_least_z{z}"] = zn["k129"], zn["k129_r200"], zn["n_at_least"]
        v[f"n_higher_z{z}"], v[f"n_tie_z{z}"] = zn["n_higher"], zn["n_tie"]
        if z == 4:
            v.update(rho_z4=zn["rho"], logp_z4=zn["logp"], ap_rho_min=zn["ap_rho_min"], ap_rho_max=zn["ap_rho_max"],
                     ap_p_max=zn["ap_p_max"])
    pk = parent_numbers(data["mdpl2_parent"], data["zooms_3k"]["z4"])
    names = dict(n_top="pk_n_top", n_rand="pk_n_rand", n_fit="pk_n_fit", slope="pk_slope", intercept="pk_icpt",
                 rho="pk_rho", scatter="pk_scatter", k129_parent="pk_k129_parent", excess="pk_excess",
                 excess_sigma="pk_excess_sig", n_other_ge="pk_n_other_ge", n_cal="pk_cal_n",
                 mean_absdiff="pk_cal_absdiff", k129_zoom3k="k129_zoom3k")
    v.update({names[k]: x for k, x in pk.items()})
    return v


# ======================================================================================================== 2. figure
def slab_image(snapshot_file, ahf_file, half_width=10.0, slab=1.5, npix=280, smooth=1.8):
    """Background of the figure: projected dark-matter density (xy) of the slab |dz| <= slab around the most massive
    AHF halo, in units of the mean of the non-empty pixels, Gaussian-smoothed by `smooth` pixels.
    Returns (image [y, x], redshift, R200c of that halo, its centre) in comoving Mpc/h."""
    import h5py
    z = float(Path(ahf_file).name.split(".z")[1].split(".AHF")[0])
    ah = np.loadtxt(ahf_file); i = np.argmax(ah[:, 3]); cen = ah[i, 5:8] / 1e3; r200 = ah[i, 11] / 1e3
    lo, hi = cen - half_width, cen + half_width
    with h5py.File(snapshot_file, "r") as h:
        pos = h["PartType1"]["Coordinates"][:] / 1e3
        p = pos[np.all((pos >= lo) & (pos < hi), axis=1)] - cen
    q = p[np.abs(p[:, 2]) <= slab]
    e = np.linspace(-half_width, half_width, npix + 1)
    H = ndimage.gaussian_filter(np.histogram2d(q[:, 0], q[:, 1], bins=[e, e])[0], smooth)
    mean = H[H > 0].mean() if np.any(H > 0) else 1.0
    return (H / mean).T, z, r200, cen


def split_filaments(pts, lens):
    """Concatenated filament points and their lengths -> list of (n, 3) arrays."""
    o = np.r_[0, np.cumsum(lens)]
    return [np.asarray(pts[o[i]:o[i + 1]], float) for i in range(len(lens))]


def draw_filament(ax, pts, slab=1.5, color="#d62728"):
    """The parts of a filament inside the slab |z| <= slab, projected on xy (a jump > 5 Mpc/h starts a new part)."""
    import matplotlib.patheffects as pe
    fx = [pe.withStroke(linewidth=2.4, foreground="w")]

    def seg(r):
        ax.plot(r[:, 0], r[:, 1], color=color, lw=1.4, alpha=0.95, zorder=5, solid_capstyle="round", path_effects=fx)
    run = []
    for k, ok in enumerate(np.abs(pts[:, 2]) <= slab):
        pt = (pts[k, 0], pts[k, 1])
        if ok and (not run or np.hypot(pt[0] - run[-1][0], pt[1] - run[-1][1]) <= 5.0):
            run.append(pt)
        else:
            if len(run) >= 2:
                seg(np.array(run))
            run = [pt] if ok else []
    if len(run) >= 2:
        seg(np.array(run))


def plot_web(data, out="plots/region129_griddisperse_web.png", fs=13):
    """Extended Data figure: cosmic web around Cluster129 at z=4.038 (left) and 5.017 (right).  Greyscale: projected
    dark-matter density of the slab; red: DisPerSE filaments at the calibrated cut; blue crosses: density maxima that
    survive the cut; circles: tracer haloes of the MDPL2 parent volume (colour: log10 Mvir); star: the progenitor;
    dashed circle: 1.5 R200c.  Coordinates are centred on the most massive AHF halo of the zoom."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    pink, cmax = "#ff1493", "#56B4E9"
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.9), sharex=True, sharey=True)
    for ax, z in zip(axes, (4, 5)):
        g = data["cluster129"][f"z{z}"]; W, S = g["attrs"]["half_width"], g["attrs"]["slab"]
        img = g["slab_image"]; lg = np.log10(img[img > 0])
        ax.imshow(np.log10(img + 1e-3), origin="lower", extent=[-W, W, -W, W], cmap="gray", aspect="equal",
                  vmin=np.percentile(lg, 3), vmax=np.percentile(lg, 97))
        shift = np.asarray(g["centre"]) - np.asarray(g["ahf_centre"])   # progenitor frame -> AHF-halo frame
        for f in split_filaments(g["filament_pts"], g["filament_lens"]):
            draw_filament(ax, f + shift, slab=S)
        mx = np.asarray(g["maxima"], float) + shift
        mx = mx[(np.abs(mx[:, 2]) <= S) & (np.abs(mx[:, 0]) <= W) & (np.abs(mx[:, 1]) <= W)]
        ax.scatter(mx[:, 0], mx[:, 1], marker="+", s=70, color=cmax, lw=1.6, zorder=7)
        lm = np.log10(g["tracer_mvir"]); size = 5 + 80 * ((lm - 10.5) / 3.5).clip(0, 1) ** 2
        sc = ax.scatter(g["tracer_dx"], g["tracer_dy"], c=lm, s=size, cmap="viridis", vmin=10.5, vmax=13.8,
                        edgecolor="k", linewidth=0.25, alpha=0.95, zorder=6)
        r = 1.5 * g["r200c_ahf"]
        ax.add_patch(Circle((0, 0), r, fill=False, ec=pink, ls="--", lw=1.6, zorder=8))
        ax.text(0, r * 1.04, r"$1.5R_{\rm 200c}$", color=pink, ha="center", va="bottom", fontsize=fs - 4, zorder=9,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="k")])
        ax.scatter([0], [0], marker="*", s=360, color=pink, edgecolor="w", lw=1.0, zorder=10)
        ax.set_title(rf"$z={g['attrs']['redshift']:.2f}$   (cut $={g['cut']:g}$,  $N_{{\rm halo}}={len(lm)}$)", fontsize=fs + 1)
        ax.set_xlabel(r"$\Delta x$ [Mpc/$h$]", fontsize=fs); ax.tick_params(labelsize=fs - 2)
        ax.set_xlim(-W, W); ax.set_ylim(-W, W)
    axes[0].set_ylabel(r"$\Delta y$ [Mpc/$h$]", fontsize=fs)
    cb = fig.colorbar(sc, ax=axes, fraction=0.023, pad=0.01)
    cb.set_label(r"$\log_{10}\,(M_{\rm vir}\,[M_\odot/h])$  of tracer halos", fontsize=fs - 2)
    cb.ax.tick_params(labelsize=fs - 3)
    leg = [Line2D([], [], color="#d62728", lw=1.8, label="DisPerSE on binned overdensity (absolute cut)"),
           Line2D([], [], marker="+", ls="none", color=cmax, ms=9, mew=1.6, label="density maxima"),
           Line2D([], [], marker="o", ls="none", mfc="#3b8", mec="k", ms=7, label="tracer halos"),
           Line2D([], [], color=pink, marker="*", ls="none", ms=15, mec="k", mew=0.9, label="region 129 node")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=4, fontsize=fs - 3)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=170, bbox_inches="tight")
    return fig


# ======================================================================================================== 3. pipeline
def read_dm_cube(snapshot_file, centre, lhalf=LHALF):
    """High-resolution dark-matter particles (PartType1) of a GIZMO snapshot inside the cube centre +- lhalf
    [comoving Mpc/h] and the particle mass [Msun/h]."""
    import h5py
    centre = np.asarray(centre, float)
    with h5py.File(snapshot_file, "r") as h:
        mdm = float(h["PartType1"]["Masses"][0]) * 1e10
        pos = h["PartType1"]["Coordinates"][:] / 1e3
    p = pos[np.all((pos >= centre - lhalf) & (pos < centre + lhalf), axis=1)]
    if len(p) == 0:
        raise RuntimeError(f"no high-resolution dark matter in the cube about {centre.tolist()}")
    return p, mdm


def cic_delta(pos, lo, ng, cell, mdm):
    """Cloud-in-cell overdensity delta = rho / rho_DM_mean - 1 on the grid nodes lo + i * cell (i = 0..ng-1).
    The upper neighbour of the last node wraps to node 0 (only particles within one cell of the upper face)."""
    g = np.zeros((ng, ng, ng), dtype=np.float64)
    x = (pos - lo) / cell; i0 = np.floor(x).astype(int); d = x - i0
    for ox in (0, 1):
        wx = (d[:, 0] if ox else 1 - d[:, 0]); ix = (i0[:, 0] + ox) % ng
        for oy in (0, 1):
            wy = (d[:, 1] if oy else 1 - d[:, 1]); iy = (i0[:, 1] + oy) % ng
            for oz in (0, 1):
                wz = (d[:, 2] if oz else 1 - d[:, 2]); iz = (i0[:, 2] + oz) % ng
                np.add.at(g, (ix, iy, iz), wx * wy * wz)
    return g * mdm / cell ** 3 / RHO_DM_MEAN - 1.0


def density_field(pos, centre, mdm, lhalf=LHALF, cell=CELL, rs=RS):
    """Smoothed overdensity of the cube centre +- lhalf: the particles inside the cube (lower faces included, upper
    excluded) deposited by CIC onto (2 lhalf / cell)^3 nodes, normalised by the cosmic mean dark-matter density,
    Gaussian of standard deviation rs (mode 'nearest'); float32, axes (x, y, z)."""
    ng = int(round(2 * lhalf / cell)); cell = 2 * lhalf / ng
    pos, centre = np.asarray(pos, float), np.asarray(centre, float)
    pos = pos[np.all((pos >= centre - lhalf) & (pos < centre + lhalf), axis=1)]
    delta = cic_delta(pos, centre - lhalf, ng, cell, mdm)
    return ndimage.gaussian_filter(delta, rs / cell, mode="nearest").astype(np.float32)


def _ctag(c):
    return f"{float(c):g}"


def run_disperse(delta, workdir, cut, disperse_bin, fresh_cut=FRESH_CUT):
    """DisPerSE on the overdensity grid with open boundaries.  The Morse-Smale complex is computed once
    (mse -periodicity 0 -cut fresh_cut -forceLoops, kept as d.fits.MSC) and simplified at the absolute persistence
    `cut` (mse -loadMSC); skelconv -breakdown -smooth 8 writes the skeleton as NDskl_ascii.  Returns its path.
    Raises if mse did not build the boundary mask (it would then treat the cube as periodic)."""
    from astropy.io import fits
    wdir = Path(workdir); wdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", LC_ALL="C")

    def run(cmd, fresh=False):
        r = subprocess.run(cmd, cwd=str(wdir), capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout or "")[-800:])
        if "Boundary detection disabled" in r.stdout or "Skipping mask" in r.stdout:
            raise RuntimeError("mse skipped the boundary mask: -periodicity 0 not honoured (use DisPerSE v0.9.25)")
        if fresh and "Building mask" not in r.stdout:
            raise RuntimeError("mse did not report building the boundary mask")
    fits.writeto(str(wdir / "d.fits"), np.asarray(delta, np.float32), overwrite=True)
    msc = wdir / "d.fits.MSC"
    if msc.exists():
        msc.unlink()
    try:
        run([f"{disperse_bin}/mse", "d.fits", "-periodicity", "0", "-cut", _ctag(fresh_cut), "-upSkl", "-forceLoops"],
            fresh=True)
    except RuntimeError:
        msc.unlink(missing_ok=True)
        raise
    for x in wdir.glob("*.NDskl"):
        x.unlink()
    run([f"{disperse_bin}/mse", "d.fits", "-loadMSC", "d.fits.MSC", "-cut", _ctag(cut), "-upSkl"])
    skl = [x for x in os.listdir(wdir) if x.endswith(".up.NDskl")]
    if not skl:
        raise RuntimeError(f"mse wrote no skeleton at cut {cut}")
    run([f"{disperse_bin}/skelconv", skl[0], "-breakdown", "-smooth", "8", "-to", "NDskl_ascii"])
    asc = sorted([x for x in os.listdir(wdir) if x.startswith(skl[0]) and x.endswith(".NDskl")], key=len)[-1]
    return wdir / asc


def parse_skeleton(path, lhalf=LHALF, cell=CELL):
    """Arcs of an NDskl_ascii skeleton in cube-centred coordinates: FITS pixel (k, j, i) -> (x, y, z) =
    -lhalf + (i, j, k) * cell.  Arcs that end at a critical point flagged as boundary are dropped (they end on the
    cube faces).  Returns (arcs, end_types (n, 2): critical-point types at the two ends, 2 = saddle, 3 = maximum,
    4 = bifurcation; maxima (m, 3): all type-3 critical points).  Raises if an arc steps more than half the cube
    (DisPerSE ran with periodic boundaries)."""
    ax0 = -lhalf

    def to_xyz(c):
        c = np.asarray(c, float).reshape(-1, 3)
        return np.stack([ax0 + c[:, 2] * cell, ax0 + c[:, 1] * cell, ax0 + c[:, 0] * cell], axis=-1)
    lines = Path(path).read_text().splitlines()
    arcs, ends, bflag, ctype, maxima, half = [], [], [], [], [], None
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if s.startswith("BBOX"):
            lo, hi = [np.array(b.split(","), float) for b in s[4:].replace("[", " ").replace("]", " ").split()]
            half = 0.5 * (hi - lo)
        if s == "[CRITICAL POINTS]":
            ncp = int(lines[i + 1]); i += 2
            for _ in range(ncp):
                rec = lines[i].split()
                ctype.append(int(float(rec[0]))); bflag.append(int(rec[-1]) != 0)
                if int(float(rec[0])) == 3:
                    maxima.append([float(rec[1]), float(rec[2]), float(rec[3])])
                i += 2 + int(lines[i + 1])
            continue
        if s == "[FILAMENTS]":
            nf = int(lines[i + 1]); i += 2
            for _ in range(nf):
                cp1, cp2, ns = (int(v) for v in lines[i].split()[:3]); i += 1
                pts = np.array([[float(v) for v in lines[i + k].split()[:3]] for k in range(ns)])
                i += ns
                if ns > 1:
                    if half is None:
                        raise RuntimeError(f"{path}: no BBOX line, cannot check for arcs that wrap the cube")
                    if np.any(np.abs(np.diff(pts, axis=0)).max(axis=0) > half):
                        raise RuntimeError(f"{path}: an arc steps more than half the cube: DisPerSE ran periodic")
                if bflag[cp1] or bflag[cp2]:
                    continue
                arcs.append(to_xyz(pts)); ends.append((ctype[cp1], ctype[cp2]))
            continue
        i += 1
    return arcs, np.asarray(ends, int).reshape(-1, 2), to_xyz(maxima) if maxima else np.zeros((0, 3))


def merge_arcs(arcs, endpoint_mergeable, max_angle=MERGE_ANGLE, tol=MERGE_TOL, n_tangent=N_TANGENT):
    """Join arcs that continue each other (open boundaries).  Endpoints closer than `tol` form a junction (at least two
    different arcs); a junction is used only if one of its endpoints is mergeable (a saddle).  At each junction the
    best-aligned pair of arcs is joined if it deviates from a straight continuation by less than `max_angle` degrees
    (tangents from `n_tangent` points).  Linked arcs form chains; a closed loop is cut at its least straight link.
    Returns (filaments, members): the joined polylines (a shared junction point kept once) and the input arcs of each,
    in order.  Every arc is in exactly one filament."""
    arcs = [np.asarray(a, float) for a in arcs]; N = len(arcs)
    if N == 0:
        return [], []
    mergeable = np.asarray(endpoint_mergeable, bool).ravel()
    if len(mergeable) != 2 * N:
        raise ValueError("endpoint_mergeable needs one flag per endpoint (start of arc i at 2i, end at 2i+1)")
    owner = np.repeat(np.arange(N), 2)
    eps = np.empty((2 * N, 3)); eps[0::2] = [a[0] for a in arcs]; eps[1::2] = [a[-1] for a in arcs]
    pairs = cKDTree(eps).query_pairs(r=tol, output_type="ndarray")
    pairs = pairs[owner[pairs[:, 0]] != owner[pairs[:, 1]]] if len(pairs) else pairs.reshape(0, 2)
    _, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(2 * N, 2 * N)),
                                  directed=False)
    lab = lab.astype(np.int64)                           # a junction needs endpoints of at least two arcs
    order = np.lexsort((owner, lab)); l_s, o_s = lab[order], owner[order]
    grp = np.cumsum(np.r_[True, l_s[1:] != l_s[:-1]]) - 1
    n_own = np.zeros(grp[-1] + 1, int)
    np.add.at(n_own, grp, np.r_[True, (l_s[1:] != l_s[:-1]) | (o_s[1:] != o_s[:-1])].astype(int))
    keep = n_own[grp] >= 2
    lab = np.full(2 * N, -1, np.int64); lab[order[keep]] = l_s[keep]
    tan = np.empty((2 * N, 3))                           # unit tangent at each endpoint, pointing into the arc
    for i, a in enumerate(arcs):
        m = min(n_tangent, len(a)); tan[2 * i] = a[m - 1] - a[0]; tan[2 * i + 1] = a[-m] - a[-1]
    nrm = np.linalg.norm(tan, axis=1); bad = nrm == 0
    tan[~bad] /= nrm[~bad, None]; tan[bad] = [1.0, 0.0, 0.0]
    partner = np.full(2 * N, -1, np.int64); link_angle = np.full(2 * N, np.nan)
    ep = np.where(lab >= 0)[0]; ep = ep[np.argsort(lab[ep], kind="stable")]
    bounds = np.r_[0, np.where(np.diff(lab[ep]) != 0)[0] + 1, len(ep)]
    for k in range(len(bounds) - 1):
        e = ep[bounds[k]:bounds[k + 1]]
        if len(e) < 2 or not mergeable[e].any():
            continue
        cosdev = -(tan[e] @ tan[e].T)
        cosdev[(e[:, None] // 2) == (e[None, :] // 2)] = -np.inf
        np.fill_diagonal(cosdev, -np.inf)
        a, b = np.unravel_index(np.argmax(cosdev), cosdev.shape)
        ang = float(np.degrees(np.arccos(np.clip(cosdev[a, b], -1.0, 1.0))))
        if np.isfinite(cosdev[a, b]) and ang < max_angle:
            partner[e[a]], partner[e[b]] = e[b], e[a]; link_angle[e[a]] = link_angle[e[b]] = ang
    visited = np.zeros(N, bool)

    def walk(ep_in):
        chain = []
        while True:
            i = ep_in // 2; visited[i] = True; chain.append((i, ep_in % 2 == 0))
            nxt = partner[ep_in ^ 1]
            if nxt < 0 or visited[nxt // 2]:
                return chain
            ep_in = nxt
    chains = []
    for i in range(N):
        if not visited[i] and (partner[2 * i] < 0 or partner[2 * i + 1] < 0):
            chains.append(walk(2 * i if partner[2 * i] < 0 else 2 * i + 1))
    for i in range(N):                                   # what is left lies on closed loops
        if visited[i]:
            continue
        loop, e_in = [], 2 * i
        while True:
            loop.append(e_in ^ 1); e_in = partner[e_in ^ 1]
            if e_in // 2 == i:
                break
        cut = max(loop, key=lambda x: (link_angle[x], -x))
        chains.append(walk(partner[cut]))
    chains.sort(key=lambda c: min(m for m, _ in c))
    filaments, members = [], []
    for c in chains:
        parts = []
        for k, (i, fwd) in enumerate(c):
            a = arcs[i] if fwd else arcs[i][::-1]
            if k and np.linalg.norm(a[0] - parts[-1][-1]) <= 1e-9 * max(1.0, float(np.abs(a[0]).max())):
                a = a[1:]
            parts.append(a)
        filaments.append(np.vstack(parts)); members.append([i for i, _ in c])
    if sorted(i for m in members for i in m) != list(range(N)):
        raise RuntimeError("merge_arcs: an arc is missing or used twice")
    return filaments, members


def smooth_filament(pts, niter=SMOOTH_ITER):
    """Neighbour averaging with the endpoints fixed, niter times.  Computed in a frame shifted by +500 Mpc/h inside a
    1000 Mpc/h periodic box (the frame of the published pipeline; the shift keeps every cube point inside the box, so
    the result is that of open boundaries)."""
    p = np.asarray(pts, float) + _OFFSET
    if len(p) > 2 and niter > 0:
        for _ in range(niter):
            new = p.copy(); cur = p[1:-1]
            dprev = p[:-2] - cur; dprev -= _BOX * np.round(dprev / _BOX)
            dnext = p[2:] - cur; dnext -= _BOX * np.round(dnext / _BOX)
            new[1:-1] = (cur + (dprev + dnext) / 2.0) % _BOX
            p = new
    return p - _OFFSET


def postprocess(arcs, end_types):
    """Post-processing of Singh et al. (2026): drop arcs with fewer than two points or shorter than 0.01 Mpc/h, join
    arcs through saddle points (merge_arcs), smooth each filament (smooth_filament)."""
    end_types = np.asarray(end_types, int).reshape(-1, 2)
    if len(end_types) != len(arcs):
        raise ValueError("one pair of end types per arc")
    arcs = [np.asarray(a, float) for a in arcs]
    keep = [k for k, a in enumerate(arcs) if len(a) >= 2 and np.linalg.norm(np.diff(a, axis=0), axis=1).sum() >= 0.01]
    arcs, end_types = [arcs[k] for k in keep], end_types[keep]
    fils, _ = merge_arcs(arcs, end_types.ravel() == 2)
    return [smooth_filament(f) for f in fils]


def count_kappa(filaments, radius, centre=(0.0, 0.0, 0.0)):
    """Connectivity: the number of filaments with at least one point inside the sphere (distance <= radius) and one
    outside it."""
    c = np.asarray(centre, float); n = 0
    for f in filaments:
        d = np.sqrt(((np.asarray(f, float) - c) ** 2).sum(axis=1))
        n += bool(np.any(d <= radius) and np.any(d > radius))
    return n


def measure_region(snapshot_file, centre, cut, disperse_bin, workdir, r200c=None):
    """The whole method for one zoom at one snapshot: cube of high-resolution dark matter around `centre`, smoothed
    overdensity, DisPerSE at the absolute persistence `cut`, post-processing and connectivity within 1 Mpc/h (and
    1.5 R200c if r200c is given).  Coordinates of the returned filaments and maxima are relative to `centre`."""
    pos, mdm = read_dm_cube(snapshot_file, centre)
    delta = density_field(pos, centre, mdm)
    del pos
    skl = run_disperse(delta, workdir, cut, disperse_bin)
    arcs, ends, maxima = parse_skeleton(skl)
    fils = postprocess(arcs, ends)
    return dict(filaments=fils, maxima=maxima, n_arcs=len(arcs), kappa_1Mpc=count_kappa(fils, 1.0),
                kappa_1p5R200=None if r200c is None else count_kappa(fils, 1.5 * r200c))
