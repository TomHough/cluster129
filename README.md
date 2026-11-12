# Cluster129: a massive protocluster at z ≈ 4

Analysis notebooks for the study of **Cluster129**, an extremely early-forming halo that already reaches
M₂₀₀c ≈ 10¹⁴ M☉ at z = 4. It is the most massive z = 4 halo of MDPL2, re-simulated as a
Gizmo-Simba zoom-in from The Three Hundred project. The notebooks compare its halo mass, cold gas, star
formation, far-IR luminosity, stellar mass and thermal SZ signal with the rest of the cluster sample, with other
large-volume simulations, and with observed z ~ 4–5 protoclusters (SPT2349-56, the Distant Red Core, GN20,
SSA22, AzTEC-3, HDF850.1, …).

> Paper in preparation. If you use this code, please cite the paper once it is available (see [Citation](#citation)).

## Contents

| Notebook | What it does | Main output |
|---|---|---|
| `halo_evolution_panels.ipynb` | Evolution of main-halo M₂₀₀c, central cold-gas mass and central SFR for Cluster129 vs. the full sample (median and 10–90th percentiles). Includes Gizmo-3k, Gadget-X and SAG versions, plus a zoom at z ≈ 3–6 with observed protocluster cores. | Fig. 1 panels |
| `Volume_comparison.ipynb` | Most massive z = 4 halo vs. simulated volume for FLAMINGO, HugeMDPL, BigMDPL, MillenniumTNG, TNG-Cluster and MDPL2 (Cluster129). | `volume_vs_mass.pdf/png` |
| `Mass_vs_redshift.ipynb` | Stellar mass of the central galaxy vs. redshift, against dusty SFGs, massive quiescent galaxies, high-z JWST galaxies and DRC / SPT2349-56 members. | `mass_vs_redshift.png` |
| `Cumulative_SFR.ipynb` | Cumulative SFR vs. projected area (in the style of Hill et al. 2020, Fig. 10), median and 10–90th percentiles over random lines of sight. | `sfr_area_mpc_random200.png` |
| `Cumulative_ColdGas.ipynb` | Enclosed cold-gas mass vs. radius at several redshifts, against SPT2349-56 and DRC measurements. | `mgas_vs_radius.png` |
| `Flux_FIR_with_obs.ipynb` | Far-IR luminosity of the central galaxy vs. redshift, against a compilation of L_FIR / L_IR for high-z protocluster cores. | `L_FIR_obs.png` |
| `tSZ-maps.ipynb` | Compton-y maps with [`pymsz`](https://github.com/weiguangcui/pymsz), integrated Y within varying apertures and comparison with the self-similar Y–M relation. | FITS maps, `tSZ_aperture_mass.png` |
| `obs_lir.py` | Literature compilation of L_IR / L_FIR (with references and derived sums) imported by `Flux_FIR_with_obs.ipynb`. Running it prints the tables and writes them to `obs_lir.h5`. | – |

## Requirements

Python ≥ 3.8 and:

```bash
pip install -r requirements.txt
```

`tSZ-maps.ipynb` also needs `pymsz`, installed from its repository:

```bash
pip install git+https://github.com/weiguangcui/pymsz.git
```

The notebooks import a plotting-style helper, `set_style` (`set_style('mnras')`), which is not included here.
Either add your own `set_style.py` next to the notebooks or remove those calls.

## Data

The simulation data are **not** included in this repository. The notebooks expect:

- **Simulation outputs** (Gizmo-Simba snapshots, Caesar and AHF halo catalogues, SAG outputs). These are read
  from absolute paths on the original cluster (`/data20/TheThreeHundred/...`). Edit the path variables at the
  top of each notebook to point to your copy.
- **Processed HDF5 files** in the working directory: `HaloID_track_Gizmo7k.hdf5`, `Historic_data_324.hdf5`,
  `Mgas_sfr_complete_*_comovil.hdf5`, `Mgas_sfr_radius_129.hdf5`, `central_gal_properties_all.hdf5`.
- **Digitised observational curves** in `obs/` (`ssa22.csv`, `gn20.csv`, `drc.csv`, `spt.csv`) for the
  cumulative SFR plot.
- Output folders `plots/` and `tsz-map/`.

Access to The Three Hundred data can be requested from the collaboration
(<https://www.nottingham.ac.uk/astronomy/The300/>).

Observational values taken from the literature are listed directly in the notebooks, with their references.
Some were digitised from published figures (noted in the code). For exact values, use the original papers.

Cosmology: flat ΛCDM with H₀ = 67.8 km s⁻¹ Mpc⁻¹, Ω_m = 0.31, h = 0.678.

## Citation

Paper in preparation. A citation entry will be added here once it is available.

## License

Released under the [MIT License](LICENSE).
