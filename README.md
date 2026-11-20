# CT-PoreFlow

Code for **CT-PoreFlow**, a continuous-time neural operator for pore-scale transient flow in two-dimensional QSGS porous media, trained on D2Q9-BGK lattice Boltzmann (LBM) simulations, together with generative inverse design (conditional GAN and diffusion) and LBM verification.

Dataset and trained checkpoints: https://huggingface.co/JonathanWWW/CT-PoreFlow

## Repository layout

- `transient_flow_dl/` - PyTorch training, evaluation and analysis pipeline
  - `src/` - data loading, models (CT-PoreFlow, Time-U-Net, Time-FNO, DeepONet, ConvLSTM, Transformer), losses and metrics
  - `inverse_design/` - conditional GAN, diffusion model, QSGS closed-loop inverse design and LBM verification
  - `configs/` - experiment configurations (main comparison, ablations, temporal hold-out, inverse design)
  - `tests/` - unit tests
- `lbm_data_generation/` - QSGS geometry generation and D2Q9 LBM solver
  - `lbm_dataset_generator.py` - stratified QSGS + steady LBM dataset generator
  - `generate_transient_snapshots.py` - transient LBM snapshot generation
  - `matlab_original/` - original MATLAB implementations (`lbm2d_cx_tecplot.m`, `qsgs2d.m`)

## Quick start

```bash
cd transient_flow_dl
pip install -r requirements.txt
# place the dataset under transient_flow_dl/data/ (see README in transient_flow_dl)
python train.py --config configs/ct_poreflow_v2_transport_seed2026.json
python evaluate.py --run-dir runs/ct_poreflow_v2_transport_seed2026
```

See `transient_flow_dl/README.md` for the full list of commands.

## Citation

If you use this code or data, please cite the CT-PoreFlow paper (citation will be added upon publication).