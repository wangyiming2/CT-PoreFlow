# Transient pore-flow learning

Reproducible PyTorch pipeline for predicting a D2Q9-BGK LBM transient from QSGS pore geometry and a continuous query time. The repository implements Time-U-Net, Time-FNO, DeepONet, a query-time ConvLSTM, a geometry-time Transformer, topology-aware variants and the proposed CT-PoreFlow steady-residual neural operator. The paper's completed inverse-design experiment uses auditable QSGS candidate generation, surrogate terminal-field screening, and unchanged-solver LBM verification. Conditional GAN and diffusion generative models are provided in `inverse_design/`.

The learning target is a numerical field tensor, not RGB video. A trained model returns normalized `ux`, `uy` and density/pressure proxy fields for any requested time; speed is derived exactly from the two velocity components. The visualization command exports the 30-frame numerical trajectory, a fixed-training-scale GIF and a six-time comparison sheet.

## Learning contract

For CT-PoreFlow, one supervised query contains:

- static input `X_G` with shape `5 x 100 x 100`: pore mask, inlet--outlet spanning-component mask, EDT divided by 100, inlet geodesic distance divided by 100, and outlet geodesic distance divided by 100;
- one scalar query `tau_t = log(1+t) / log(50001)`;
- target `Y(t)` with shape `3 x 100 x 100`: `ux / s_u`, `uy / s_u`, and `(rho-rho_out)/(rho_in-rho_out)`.

Here `s_u=5.024437996326014e-06` is the training-only pore-speed q99.5 scale. Geodesic values may exceed one for graph paths longer than the lattice width. The spanning-component mask includes attached cul-de-sac branches; it is not a skeletonized transport backbone. Every predicted channel is hard-masked to zero in solid nodes.

Speed, flux, Darcy velocity, permeability, hydraulic tortuosity, divergence, pressure drop, mass balance and solid leakage are derived from the inverse-normalized prediction. They are evaluation quantities rather than additional network outputs. RGB colors, GIFs and MP4s are display products only.

## Data layout (relative to transient_flow_dl)

- Steady geometry/labels: `data\LBM_8000_20260714_dataset`
- Transient fields: `data\LBM_Transient_30Step`

## Commands

```powershell
python audit_dataset.py --config configs\unet_baseline.json
python compute_normalization.py --config configs\unet_baseline.json
python train.py --config configs\smoke.json
python train.py --config configs\unet_baseline.json
python train.py --config configs\ct_poreflow_v2_transport_seed2026.json
python evaluate.py --run-dir runs\unet_time_seed2026
python visualize_predictions.py --run-dir runs\unet_time_seed2026
python visualize_continuous_animation.py --run-dir runs\ct_poreflow_seed2026 --frames 60
python analyze_results.py --run-dir runs\unet_time_seed2026
python stratified_analysis.py --run-dir runs\unet_time_seed2026
python plot_efficiency_tradeoff.py --run runs\unet_time_seed2026 --label Time-U-Net --run runs\fno_time_seed2026 --label Time-FNO --run runs\deeponet_seed2026 --label DeepONet --run runs\ct_poreflow_seed2026 --label CT-PoreFlow --output analysis\efficiency_tradeoff.png
python plot_training_curves.py --run runs\unet_time_seed2026 --label Time-U-Net --run runs\fno_time_seed2026 --label Time-FNO --run runs\deeponet_seed2026 --label DeepONet --run runs\ct_poreflow_seed2026 --label CT-PoreFlow --output analysis\training_curves.png
python train.py --config configs\unet_temporal_holdout.json
python evaluate.py --run-dir runs\unet_temporal_holdout_seed2026 --time-indices 2 4 7 9 12 14 17 19 22 27 --evaluation-name temporal_holdout
python temporal_holdout_analysis.py --baseline-run runs\unet_temporal_holdout_seed2026 --proposed-run runs\ct_poreflow_temporal_holdout_seed2026 --output-dir analysis\temporal_holdout
python select_representative_samples.py
python plot_multimodel_mosaics.py --bundle-dir ..\output\representative_samples\bundles
python select_extended_gallery_samples.py --topology ..\output\representative_samples\all_test_topology_metrics.csv --unet ..\output\formal_results\unet_time_seed2026\analysis\stratified\sample_level_metrics.csv --ct ..\output\formal_results\ct_poreflow_seed2026\analysis\stratified\sample_level_metrics.csv --output ..\output\extended_prediction_gallery\selected_samples_16.json
python export_multimodel_representatives.py --run Time-U-Net=runs\unet_time_seed2026 --run Time-FNO=runs\fno_time_seed2026 --run DeepONet=runs\deeponet_seed2026 --run CT-PoreFlow=runs\ct_poreflow_seed2026 --sample-indices 8,158,226,341,423,525,646,697,144,201,143,702,163,16,10,206 --output runs\extended_prediction_gallery
python plot_extended_prediction_gallery.py --bundle-dir ..\output\extended_prediction_gallery\bundles --selection ..\output\extended_prediction_gallery\selected_samples_16.json --output ..\output\extended_prediction_gallery\figures
python train.py --config configs\convlstm_baseline_seed2026.json
python train.py --config configs\transformer_baseline_seed2026.json
python -m inverse_design.train --config configs\inverse_cgan_seed2026.json
python -m inverse_design.train --config configs\inverse_diffusion_seed2026.json
python -m inverse_design.sample --run-dir runs\inverse_diffusion_seed2026 --targets 0.55,0.50,0.09,0.70,1.30
python inverse_design\qsgs_closed_loop.py --skip-novel --output-root ..\output\inverse_design
python inverse_design\qsgs_closed_loop.py --novel-candidates 120 --output-root inverse_results_novel
python inverse_design\analyze_closed_loop.py --input ..\output\inverse_design\offline_inverse_design_selections.csv --novel ..\output\inverse_design\novel_qsgs_lbm_verification.csv --output-dir ..\output\inverse_design
```

Do not use the per-sample normalized steady fields as training targets. Dynamic `ux`, `uy`, and `rho` are transformed with training-split-only statistics and fixed physical pressure scaling.

## Registered formal experiments

Main comparison:

- `configs/unet_baseline.json`: sinusoidal-time-conditioned U-Net.
- `configs/fno_baseline.json`: Fourier neural operator with coordinates and time map.
- `configs/deeponet_baseline.json`: convolutional branch and `(x,y,t)` trunk.
- `configs/proposed_baseline.json`: CT-PoreFlow with five geometry channels, spectral bottleneck and steady-anchor transient-residual decoding.
- `configs/ct_poreflow_v2_transport_seed2026.json`: validation-selected CT-PoreFlow v2. It retains the five topology channels, continuous-time decoder and spectral bottleneck, removes the harmful hard terminal residual cutoff, and fine-tunes with a late-time inlet/outlet-flux loss. Early nonzero and terminal frames are modestly oversampled. The run is initialized from the no-anchor ablation checkpoint and checkpoint selection still uses validation loss only.

Controlled ablations:

- `ablation_topology_unet.json`
- `ablation_no_spectral.json`
- `ablation_no_topology.json`
- `ablation_no_steady_anchor.json`
- `ablation_linear_time.json`
- `ablation_data_loss_only.json`

Dedicated unseen-time interpolation:

- `unet_temporal_holdout.json`
- `ct_temporal_holdout.json`

These runs train on 20 declared stored-time indices, keep 10 interleaved indices out of both training and checkpoint selection, and evaluate the latter on all 764 test geometries.

Each formal configuration uses the same geometry-level split, training-only velocity scale, optimizer family, epoch count and fixed validation time indices.

## Output contract

Each `runs/<experiment>/` directory contains:

- `config.json`, `environment.json`, `model.json` and normalization metadata;
- `best.pt`, `last.pt`, periodic checkpoints and `history.jsonl`;
- full test-set `per_sample_time_metrics.csv` and aggregate `summary.json`;
- per-time field and physics tables/figures;
- sample-level topology-stratified analysis and worst-case lists;
- a prediction GIF, six-time contact sheet and compressed numerical trajectory NPZ.

The CT-PoreFlow postprocess also emits a 60-query continuous-time trajectory. Intermediate queries are explicitly marked as interpolation because direct LBM labels exist only at the original 30 stored times. RGB GIF/MP4 files are derived display products; the NPZ numerical tensors remain the scientific and learning artifacts.

Cross-run tools produce the performance table, time-resolved comparison, formal 50-epoch convergence curves, accuracy--efficiency Pareto plot, physical calibration, bootstrap confidence intervals, paired Wilcoxon tests, matched rank-biserial effects, and Holm--Bonferroni family-wise-error correction. `benchmark_warm_start.py` reconstructs D2Q9 equilibrium populations from the predicted final field and continues the unchanged LBM solver until the original convergence criteria are satisfied, including the explicit maximum-Mach gate. The paired warm-start analysis also reports the converged permeability difference relative to the matched rest-state Python-LBM solve, so computational savings cannot be separated from solution consistency.

The extended qualitative audit uses 16 deterministically selected test geometries rather than visually hand-picked examples. It includes one typical-error case from each porosity bin, morphology and permeability extremes, the easiest and hardest joint cases, and the largest model-wise advantage for Time-U-Net and CT-PoreFlow. `plot_extended_prediction_gallery.py` produces two terminal truth/prediction/vector-error galleries and two four-time transient galleries in PNG, vector PDF, and editable SVG. All speed fields retain the training-derived fixed scale; terminal error maps share one declared gallery-wide scale.

## Rule-based morphology OOD benchmark

`ood_validate.py` creates regular solid obstacles absent from QSGS training: circular arrays, snowflakes, tree-like branching obstacles, and staggered baffles. Target porosity is matched within the training range, while the first and last flow rows remain open. The `generate` stage writes deterministic geometry masks and the `lbm` stage runs the unchanged D2Q9-BGK solver for 30 logarithmic snapshots. The `evaluate` stage loads frozen checkpoints and reports per-frame and terminal relative velocity errors for all four model families.

```powershell
python ood_validate.py --stage generate --porosities 0.60 0.68 --samples-per-condition 3
python ood_validate.py --stage lbm
python ood_validate.py --stage evaluate --run-dir runs\unet_time_seed2026 --run-dir runs\fno_time_seed2026 --run-dir runs\deeponet_seed2026 --run-dir runs\ct_poreflow_seed2026
```

The generated benchmark is stored under `output/ood_rule_validation`. It includes the numerical LBM tensors, convergence metadata, family-level summaries, and PNG/PDF/SVG geometry and model-comparison figures. The reported OOD run is frozen-model evaluation; no OOD fine-tuning or sample-specific normalization is applied.

## Closed-loop inverse design used in the paper

`inverse_design/qsgs_closed_loop.py` provides two complementary experiments. The offline experiment builds 18 target specifications only from training-split quantiles and compares Random-QSGS, morphology-only retrieval, Time-U-Net, CT-PoreFlow, the no-anchor ablation, and a labelled LBM-oracle bound on the 764 unseen test geometries. The novel-candidate experiment generates a new seeded pool, predicts terminal transport with Time-U-Net, selects guided and random controls, and reruns the original D2Q9-BGK solver. Outputs include complete CSV rows, candidate masks and numerical fields in NPZ, LBM quality-control values, run manifests, and PNG/PDF/SVG figures.

`inverse_design/analyze_closed_loop.py` averages random draws within target before paired inference, computes 20,000 target-level bootstrap resamples, exact two-sided sign-flip tests over all $2^{18}$ assignments, and Holm correction over the four declared comparisons. The three novel target pairs are deliberately reported as descriptive physical round trips rather than assigned an underpowered significance test.

After the remote pipeline completes, `paper_workspace/scripts/stage_manuscript_after_results.ps1` synchronizes the evidence, validates the 30- and 60-frame numerical trajectories, produces CT and four-model GIF/MP4 files, generates a SHA-256 reproducibility manifest, builds the measured-results facts file, drafts the data-aware result sections, assembles the Spark-to-Paper work directory, and runs the official draft and citation gates. It refuses to proceed if any formal run, figure, table, statistic, warm-start result, dynamic artifact, epoch history, or test-scope check is missing.

## Reproducibility and leakage controls

- All 30 time frames from one geometry remain in one split.
- EDT and geodesic channels are computed from geometry only.
- Velocity scaling and visualization limits are estimated from the training split only.
- The zero-velocity initial frame is retained for absolute error but excluded from relative-error denominators.
- Predictions are hard-masked to zero in solid nodes.
- LBM warm-start acceptance uses the original pressure boundaries, collision model and stopping criteria.
