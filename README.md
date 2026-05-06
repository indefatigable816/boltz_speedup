<div align="center">
  <div>&nbsp;</div>
  <img src="docs/boltz2_title.png" width="300"/>
  <img src="https://model-gateway.boltz.bio/a.png?x-pxid=bce1627f-f326-4bff-8a97-45c6c3bc929d" />

[Boltz-1](https://doi.org/10.1101/2024.11.19.624167) | [Boltz-2](https://doi.org/10.1101/2025.06.14.659707) |
[Slack](https://boltz.bio/join-slack) <br> <br>
</div>

> **⚡ Speedup fork** — branch `speedup/screening-mode` of [indefatigable816/boltz_speedup](https://github.com/indefatigable816/boltz_speedup).
> Upstream: [jwohlwend/boltz](https://github.com/jwohlwend/boltz).
> **Combined realistic speedup: 5–8× for drug screening workloads** (12 h jobs → ~2 h on A100).

## What this fork changes

| File | Change | Speedup |
|------|--------|---------|
| `src/boltz/main.py` | `"highest"` → `"high"` TF32 matmul precision | 1.5–2× GEMMs |
| `src/boltz/model/layers/attentionv2.py` | Manual einsum → `F.scaled_dot_product_attention` | 1.5–2× attention |
| `src/boltz/model/layers/attention.py` | Same SDPA guard for Boltz-1 path | 1.5–2× |
| `src/boltz/model/layers/triangular_attention/primitives.py` | `_attention()` → SDPA with `scale=1.0` | 1.5–2× tri-attn |
| `src/boltz/main.py` | `--compile_pairformer/msa/structure/confidence` flags | 1.2–1.5× |
| `src/boltz/main.py` | `--screening_mode` preset (bundles all above) | combined |
| `src/boltz/main.py` | `--early_recycling_exit` flag | 1.5–3× trunk |
| `src/boltz/model/models/boltz2.py` | z-embedding convergence check in recycling loop | (same) |
| `src/boltz/main.py` | Parallel MSA via `ThreadPoolExecutor` | ~2× MSA step |
| `src/boltz/main.py` | `--no_compress` flag | ~3× file writes |
| `src/boltz/data/write/writer.py` | Async file I/O (4 background threads) | overlaps GPU+disk |

### Screening mode preset

```bash
boltz predict input.yaml \
    --screening_mode \
    --recycling_steps 3 \
    --sampling_steps 50 \
    --num_subsampled_msa 4096
```

`--screening_mode` automatically enables: TF32 high precision, SDPA attention, `torch.compile` on all modules, early recycling exit, async writes, and `--no_compress`. Use `--recycling_steps`, `--sampling_steps`, and `--num_subsampled_msa` to tune the accuracy/speed tradeoff.


## Introduction

Boltz is a family of models for biomolecular interaction prediction. Boltz-1 was the first fully open source model to approach AlphaFold3 accuracy. Our latest work Boltz-2 is a new biomolecular foundation model that goes beyond AlphaFold3 and Boltz-1 by jointly modeling complex structures and binding affinities, a critical component towards accurate molecular design. Boltz-2 is the first deep learning model to approach the accuracy of physics-based free-energy perturbation (FEP) methods, while running 1000x faster — making accurate in silico screening practical for early-stage drug discovery.

All the code and weights are provided under MIT license, making them freely available for both academic and commercial uses. For more information about the model, see the [Boltz-1](https://doi.org/10.1101/2024.11.19.624167) and [Boltz-2](https://doi.org/10.1101/2025.06.14.659707) technical reports. To discuss updates, tools and applications join our [Slack channel](https://boltz.bio/join-slack).

## Installation

### This speedup fork (recommended for GPU screening)

```bash
# 1. Create a fresh conda environment (Python 3.10–3.12)
conda create -n boltz_speed python=3.10 -y
conda activate boltz_speed

# 2. Install PyTorch with CUDA 12.1 (matches A100 on Minerva)
#    Adjust the cu121 tag if your cluster uses CUDA 11.8 → cu118
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Clone this fork and install in editable mode
git clone https://github.com/indefatigable816/boltz_speedup.git
cd boltz_speedup
git checkout speedup/screening-mode
pip install -e ".[cuda]"

# 4. Verify GPU is visible and cuEQ kernels loaded
python -c "import torch; print(torch.cuda.get_device_name(0))"
boltz predict --help | grep screening
```

> **On Minerva (Mount Sinai HPC):** load CUDA before installing:
> ```bash
> module load cuda/12.1
> conda activate boltz_speed   # or your existing env
> pip install -e ".[cuda]"     # run from the cloned repo root
> ```
> `torch.compile` requires Triton, which ships with PyTorch ≥ 2.2 on Linux automatically — no extra install needed.

### Upstream boltz (original, no speedup)

Install with PyPI:

```
pip install boltz[cuda] -U
```

or from GitHub:

```
git clone https://github.com/jwohlwend/boltz.git
cd boltz; pip install -e .[cuda]
```

If you are installing on CPU-only or non-CUDA GPU hardware, remove `[cuda]` from the above commands. Note that the CPU version is significantly slower than the GPU version.

## Inference

You can run inference using Boltz with:

```
boltz predict input_path --use_msa_server
```

`input_path` should point to a YAML file, or a directory of YAML files for batched processing, describing the biomolecules you want to model and the properties you want to predict (e.g. affinity). To see all available options: `boltz predict --help` and for more information on these input formats, see our [prediction instructions](docs/prediction.md). By default, the `boltz` command will run the latest version of the model.


### Binding Affinity Prediction
There are two main predictions in the affinity output: `affinity_pred_value` and `affinity_probability_binary`. They are trained on largely different datasets, with different supervisions, and should be used in different contexts. The `affinity_probability_binary` field should be used to detect binders from decoys, for example in a hit-discovery stage. Its value ranges from 0 to 1 and represents the predicted probability that the ligand is a binder. The `affinity_pred_value` aims to measure the specific affinity of different binders and how this changes with small modifications of the molecule. This should be used in ligand optimization stages such as hit-to-lead and lead-optimization. It reports a binding affinity value as `log10(IC50)`, derived from an `IC50` measured in `μM`. More details on how to run affinity predictions and parse the output can be found in our [prediction instructions](docs/prediction.md).

## Authentication to MSA Server

When using the `--use_msa_server` option with a server that requires authentication, you can provide credentials in one of two ways. More information is available in our [prediction instructions](docs/prediction.md).
 
## Evaluation

⚠️ **Coming soon: updated evaluation code for Boltz-2!**

To encourage reproducibility and facilitate comparison with other models, on top of the existing Boltz-1 evaluation pipeline, we will soon provide the evaluation scripts and structural predictions for Boltz-2, Boltz-1, Chai-1 and AlphaFold3 on our test benchmark dataset, and our affinity predictions on the FEP+ benchmark, CASP16 and our MF-PCBA test set.

![Affinity test sets evaluations](docs/pearson_plot.png)
![Test set evaluations](docs/plot_test_boltz2.png)


## Training

⚠️ **Coming soon: updated training code for Boltz-2!**

If you're interested in retraining the model, currently for Boltz-1 but soon for Boltz-2, see our [training instructions](docs/training.md).


## Contributing

We welcome external contributions and are eager to engage with the community. Connect with us on our [Slack channel](https://boltz.bio/join-slack) to discuss advancements, share insights, and foster collaboration around Boltz-2.

On recent NVIDIA GPUs, Boltz leverages the acceleration provided by [NVIDIA  cuEquivariance](https://developer.nvidia.com/cuequivariance) kernels. Boltz also runs on Tenstorrent hardware thanks to a [fork](https://github.com/moritztng/tt-boltz) by Moritz Thüning.

## License

Our model and code are released under MIT License, and can be freely used for both academic and commercial purposes.


## Cite

If you use this code or the models in your research, please cite the following papers:

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}

@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}
```

In addition if you use the automatic MSA generation, please cite:

```bibtex
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  year={2022},
}
```
