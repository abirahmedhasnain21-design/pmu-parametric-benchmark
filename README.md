# A Parametric PMU Benchmark for ML-Based Transmission Line Fault Classification

> **Camera-ready version (ICECE 2026).** The published paper uses
> `generator_v3.py`, which uses one fault-resistance convention for every
> fault type (Rf in series with each faulted phase, ground bolted). Table IV,
> Table V (detection), Section IV-B (transfer) and the regime-shifted splits
> come from `cr_experiments.py` (run `python cr_experiments.py`; results go to
> `results/cr_results.json`). The Mendeley and Kaggle rows of Table IV come from
> `cr_e6_other.py`. The EMT agreement in Section III-C (max 0.67%, mean 0.34%)
> comes from `emt_reference_check.py`, using the Simscape runs `emt_<CLASS>.mat`
> included here. `emt_grid_check.py` compares the generator with Simscape over a
> grid of operating points (not yet run; the paper reports one point). Where this
> note and the older text below disagree, this note is right. The older
> notebooks below reproduce the submitted version, which used `generator_v2.py`.
>
> **Data is not included.** Put these next to the scripts: `PMU_Fault_Data.csv`
> (Dryad, doi:10.5061/dryad.x69p8czq3), `PMU_data.xlsx` (Mendeley Data,
> doi:10.17632/ydcz98ky28.1) and `classData.csv` (Kaggle,
> esathyaprakash/electrical-fault-detection-and-classification).

Code and data for the paper. Everything reported in it can be regenerated from
the two notebooks below.

Two notebooks rather than one, because they were written at different points and
serve different purposes. The pipeline notebook reproduces the audited work and
builds the generator; the paper-fixes notebook produces the figures that
required a change of protocol or of feature representation after the pipeline
notebook was already running. Where the two disagree, **the paper-fixes
notebook is the one the paper reports**; see the mapping table.

---

## Contents

```
PMU_fault_full_pipeline_v8.ipynb   reproduction, audit, generator, robustness, latency
PMU_paper_fixes_v2.ipynb           difficulty diagnostic (split by run), cross-dataset
                                   transfer, sampling defect, LLL/LLLG separability

generator_v2.py                    the parametric fault-waveform generator
gen_build.py                       PMU emulation and the run sweep
partF_core.py                      uniform resampling, sliding phasors, sequence features
partF_build.py                     loads the released CSV, builds both datasets

```

The audited dataset is **not redistributed here**. Download
`PMU_Fault_Data.csv` from Dryad (doi:10.5061/dryad.x69p8czq3) and place it
beside the notebooks, or pass an explicit path to `partF_build.load_base()`.

Requirements: numpy, pandas, scikit-learn, matplotlib, scipy, shap. All are
preinstalled on Colab; nothing needs `pip install` there. The EMT
cross-validation cell uses the `emt_<CLASS>.mat` files exported from the
Simulink model, which are included.

---

## Where each number in the paper comes from

| Paper | Produced by |
|---|---|
| Table I — reproduction vs reported | pipeline, cells up to the 80/20 PCA fit |
| Table II — scaling sensitivity | pipeline, cell 12 (scaled) and cell 14 (raw, and Δ) |
| Table III — swept parameters | `generator_v2.py`, `build_param()` in `gen_build.py` |
| **Table IV — difficulty diagnostic** | **paper-fixes, Fix 1** |
| §II-A — placeholder rows | pipeline, cleaning cell |
| §II-A — non-uniform sampling | **paper-fixes, Fix 4** |
| §II-C — single-feature and regime-shifted split | pipeline, diagnostic cell |
| §III-C — parametric and identity checks | `generator_v2.py`, run it as a script |
| §III-C — EMT agreement (1.1 %, 0.41 %, 1.49 %) | pipeline, cell 51 |
| §III-C — LLL/LLLG separability | **paper-fixes, Fix 3** |
| **§IV-B — cross-dataset transfer** | **paper-fixes, Fix 2** |
| §IV-C — perturbation robustness | pipeline, cells 45 and 49 |
| §IV-D — detection latency | pipeline, latency cell |

Two rows are worth stating plainly. Table IV and §IV-B **supersede** the
corresponding pipeline-notebook cells:

* Table IV. The pipeline notebook splits the parametric dataset at random over
  samples, which gives 98.77 / 50.32 / 32.63. The paper splits it so that each
  simulation run falls entirely on one side, which gives 76.29 / 42.89 / 28.25.
  The second is correct: §II-C of the paper identifies the random split as a
  leakage failure, so applying it to our own benchmark would not do.
* §IV-B. The pipeline notebook normalises each dataset by its own no-fault
  class, under which a transferred model labels everything no-fault and scores
  16.67 %. The paper rebuilds both datasets in a reference-invariant sequence
  representation instead. The 16.67 % figure is discussed in the paper as the
  artefact it is, not as a result.

---

## Figure files and figure numbers do not match

The filenames carry an older numbering and one figure was cut for space, so:

| Paper | File | Emitted by |
|---|---|---|
| Fig. 1 | `fig2_shap.pdf` | pipeline, cell 49 (cell 45 emits an earlier version) |
| Fig. 2 | `fig4_network_waveform.pdf` | pipeline, cell 47 |
| Fig. 3 | `fig6_perturbation.pdf` | pipeline, cell 49 (cell 45 emits an earlier version) |
| Fig. 4 | `fig7_latency.pdf` | pipeline, cell 45 |

Cell 49 runs after cell 45 and overwrites `fig2_shap.pdf` and
`fig6_perturbation.pdf`. Both cells now draw the perturbation figure at the
same single-column geometry, so it does not matter which ran last, but cell 49
is the authoritative one.

`fig3_protocol.pdf`, `fig5_transfer.pdf` and `fig8_emt_overlay.pdf` are emitted
but unused. The EMT overlay was cut for space; the agreement it showed is
reported numerically in §III-C, and cell 51 still produces it.

---

## The generator

`generator_v2.py` is self-validating. Run it directly:

```
python generator_v2.py
```

It executes the five parametric checks and the identity check described in
§III-C, plus checks on two parameters the paper does not sweep, and asserts on
failure. Nothing else should be run before it passes.

The paper's results use the five swept parameters of Table III. The file also
implements two further axes, off by default:

* `phase_rot` in {0,1,2} rotates which phases carry the fault. The paper's
  configuration fixes LG on A and both LL and LLG on B–C; the audited dataset
  uses A, A–B and C–A respectively.
* `meas` in {'bus','fault'} places the voltage measurement at the sending bus
  or at the fault point. The paper measures at the bus, where a line relay
  sits; the audited dataset behaves as though instrumented at the fault.

Both are described in the paper's limitations. They are **not** enabled for any
reported result, because sweeping them leaves the reverse transfer unchanged
once LLL and LLLG are merged, so they do not explain the residual gap. To
reproduce that check, call `build_param(..., sweep_phase=True)`.

`np.roll` is skipped for LLL, LLLG and NF: rolling a balanced positive-sequence
set is a 120-degree rotation rather than a relabelling of A/B/C, and applying
it would change the phase magnitudes after superposition.

---

## Known issues

* The pipeline notebook contains one cell using the IPython `!pip install`
  magic. It is valid in Jupyter and Colab, not in a plain Python interpreter.
* MLP on raw features is unstable across runs despite a fixed seed. This is
  stated in the paper rather than explained away.
* Table II's Δ column is computed from unrounded accuracies, so it will not
  always equal the difference of the two rounded columns printed beside it.
  This is noted in the table caption.
* The transfer directions in §IV-B carry a standard deviation of several points
  across generator seeds, which is why they are reported with a spread. Single
  seeds should not be quoted.
