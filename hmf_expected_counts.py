#!/usr/bin/env python3
"""
Expected number of massive haloes from several halo mass functions (HMFs).

Two quantities are computed for every HMF, threshold mass and mass definition:

  1. Mean number in a periodic cubic box of side L (default 1 Gpc/h) at fixed z:
         <N> = n(>M, z) * L^3            (L in Mpc/h, n in h^3 Mpc^-3)
         P(>=1) = 1 - exp(-<N>)          (Poisson statistics)

  2. Full-sky light-cone count between z1 and z2:
         <N> = Int_{z1}^{z2} n(>M, z) dV_c/dz dz ,   dV_c/dz = 4 pi dV_c/dz/dOmega

HMFs available (all through the `hmf` package):
  tinker08, tinker10, watson13, behroozi13, bocquet16 (hydro), despali16

  * Despali+16 is not shipped in `hmf`; it has the Sheth-Tormen functional form,
    so it is built with hmf's SMT class using the general (Delta- and z-dependent)
    parameter rescaling of Despali et al. (2016), Eq. 12, with
    x = log10(Delta / Delta_vir(z)).
  * Fits that were not calibrated on the requested mass definition (e.g. Behroozi+13
    is calibrated on M_vir) are converted by hmf assuming an NFW profile with the
    Duffy+08 concentration; these rows are flagged "(conv.)".
  * Yung+24 is not included: it is only valid for 6 <= z <= 19.

Units: halo masses are given in Msun (default) or Msun/h (--mass-unit hMsun).

Examples
--------
  python hmf_expected_counts.py --mass 1e14 --mdef 200c
  python hmf_expected_counts.py --mass 1e13 1e14 --mdef vir --cosmo planck2016
  python hmf_expected_counts.py --mass 1e13 --mass-unit hMsun --zbox 4.04
"""
import argparse
import copy
import warnings

import numpy as np
import hmf                                   # noqa: F401  (imported first, as requested)
from hmf import MassFunction
from hmf.mass_function import fitting_functions as ff
from scipy.integrate import simpson
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Cosmologies (flat LCDM).  "planck2022" = Planck 2018 legacy parameters
# (Planck Collab. 2020, A&A 641, A6); "planck2016" = Planck 2015 (A&A 594, A13).
# ----------------------------------------------------------------------------
COSMOLOGIES = {
    "planck2022": dict(H0=67.36, Om0=0.3153, Ob0h2=0.02237, sigma8=0.8111, ns=0.9649),
    "planck2016": dict(H0=67.74, Om0=0.3089, Ob0h2=0.02230, sigma8=0.8159, ns=0.9667),
}

MDEF = {"vir": "SOVirial", "200c": "SOCritical", "200m": "SOMean"}   # hmf names
FIT_LABELS = {
    "tinker08": "Tinker+08",
    "tinker10": "Tinker+10",
    "watson13": "Watson+13",
    "behroozi13": "Behroozi+13",
    "bocquet16": "Bocquet+16 (hydro)",
    "despali16": "Despali+16",
}


def despali_params(z, mdef, cosmo):
    """Despali et al. (2016) Eq. 12: Sheth-Tormen-like parameters for any Delta."""
    xb = cosmo.Om(z) - 1.0
    delta_vir = 18 * np.pi**2 + 82 * xb - 39 * xb**2          # Bryan & Norman 1998 (wrt rho_crit)
    delta = {"vir": delta_vir, "200c": 200.0, "200m": 200.0 * cosmo.Om(z)}[mdef]
    x = np.log10(delta / delta_vir)
    return dict(A=-0.1362 * x + 0.3292,
                a=0.4332 * x**2 + 0.2263 * x + 0.7665,
                p=-0.1151 * x**2 + 0.2554 * x + 0.2488)


def despali_model(mdef):
    """SMT-form HMF whose declared calibration definition is the requested one.

    Eq. 12 already rescales the parameters to Delta(mdef), so hmf must NOT apply its own
    (virial -> mdef) mass conversion on top; declaring the matching definition avoids it.
    """
    sim = copy.copy(ff.SMT.sim_definition)
    sim.halo_overdensity = mdef                       # 'vir', '200c' or '200m'
    return type(f"Despali16_{mdef}", (ff.SMT,), {"sim_definition": sim})


def fit_spec(fit, z, mdef, cosmo):
    """Return (hmf model name or class, hmf_params) for a fit key."""
    if fit == "despali16":
        return despali_model(mdef), despali_params(z, mdef, cosmo)
    if fit == "bocquet16":
        return ("Bocquet200mHydro" if mdef == "200m" else "Bocquet200cHydro"), {}
    return {"tinker08": "Tinker08", "tinker10": "Tinker10",
            "watson13": "Watson", "behroozi13": "Behroozi"}[fit], {}


_cache = {}


def cumulative_hmf(fit, z, mdef, cosmo, par):
    """Return (log10 M [Msun/h], n(>M) [h^3 Mpc^-3], converted_flag) for a fit at redshift z."""
    key = (fit, round(z, 6), mdef)
    if key in _cache:
        return _cache[key]
    model, params = fit_spec(fit, z, mdef, cosmo)
    for converted in (False, True):          # first native definition, then hmf mass conversion
        try:
            mf = MassFunction(z=z, Mmin=10.0, Mmax=16.5, dlog10m=0.01,
                              hmf_model=model, hmf_params=params,
                              mdef_model=MDEF[mdef], cosmo_model=cosmo,
                              sigma_8=par["sigma8"], n=par["ns"], transfer_model="EH",
                              disable_mass_conversion=not converted)
            out = (np.log10(mf.m), np.array(mf.ngtm), converted)
            _cache[key] = out
            return out
        except ValueError as err:
            if converted:
                raise
            if "mass definition" not in str(err):
                raise
    raise RuntimeError("unreachable")


def n_above(fit, z, M_hMsun, mdef, cosmo, par):
    """Cumulative comoving number density above M [Msun/h], in h^3 Mpc^-3."""
    logm, ngtm, conv = cumulative_hmf(fit, z, mdef, cosmo, par)
    return np.interp(np.log10(M_hMsun), logm, ngtm), conv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mass", type=float, nargs="+", default=[1e14], help="threshold mass(es)")
    ap.add_argument("--mass-unit", choices=["Msun", "hMsun"], default="Msun",
                    help="Msun (default) or Msun/h")
    ap.add_argument("--mdef", choices=list(MDEF), default="200c", help="halo mass definition")
    ap.add_argument("--cosmo", choices=list(COSMOLOGIES), default="planck2022")
    ap.add_argument("--fits", nargs="+", default=list(FIT_LABELS), choices=list(FIT_LABELS))
    ap.add_argument("--zbox", type=float, default=4.0, help="redshift of the simulation box")
    ap.add_argument("--box", type=float, default=1000.0, help="box side in Mpc/h")
    ap.add_argument("--zmin", type=float, default=3.5)
    ap.add_argument("--zmax", type=float, default=4.5)
    ap.add_argument("--nz", type=int, default=33, help="z-grid points (odd) for the light-cone integral")
    args = ap.parse_args()

    par = COSMOLOGIES[args.cosmo]
    h = par["H0"] / 100.0
    cosmo = FlatLambdaCDM(H0=par["H0"], Om0=par["Om0"], Ob0=par["Ob0h2"] / h**2, Tcmb0=2.7255)

    zgrid = np.linspace(args.zmin, args.zmax, args.nz)
    dVdz = cosmo.differential_comoving_volume(zgrid).to(u.Mpc**3 / u.sr).value * 4 * np.pi  # Mpc^3
    v_shell = (cosmo.comoving_volume(args.zmax) - cosmo.comoving_volume(args.zmin)).to(u.Mpc**3).value
    v_box = args.box**3                                                                      # (Mpc/h)^3

    print(f"Cosmology: {args.cosmo}  (H0={par['H0']}, Om0={par['Om0']}, sigma8={par['sigma8']}, ns={par['ns']})")
    print(f"Mass definition: M{args.mdef}   Box: ({args.box:g} Mpc/h)^3 at z={args.zbox}")
    print(f"Light cone: full sky, {args.zmin} < z < {args.zmax},  V_shell = {v_shell:.4e} Mpc^3")

    for M in args.mass:
        M_hMsun = M if args.mass_unit == "hMsun" else M * h
        M_msun = M_hMsun / h
        print(f"\n=== M{args.mdef} > {M:.3e} {args.mass_unit}  ({M_msun:.4e} Msun = {M_hMsun:.4e} Msun/h) ===")
        print(f"{'HMF':20s} {'n(z_box) [Mpc^-3]':>18s} {'<N> box':>11s} {'P(>=1) box':>11s} "
              f"{'N full sky':>11s} {'P(>=1) sky':>11s}")
        for fit in args.fits:
            n_box_h3, conv = n_above(fit, args.zbox, M_hMsun, args.mdef, cosmo, par)   # h^3 Mpc^-3
            n_box = n_box_h3 * h**3                                                    # Mpc^-3
            N_box = n_box_h3 * v_box
            n_z = np.array([n_above(fit, z, M_hMsun, args.mdef, cosmo, par)[0] for z in zgrid]) * h**3
            N_sky = simpson(n_z * dVdz, x=zgrid)
            tag = " (conv.)" if conv else ""
            print(f"{FIT_LABELS[fit] + tag:20s} {n_box:18.3e} {N_box:11.4g} {1 - np.exp(-N_box):11.3%} "
                  f"{N_sky:11.4g} {1 - np.exp(-N_sky):11.1%}")


if __name__ == "__main__":
    main()
