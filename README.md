# PokeFA-SDXL-LoRA

<p align="center">
  <img src="assets/welcome_pikachu.png" alt="Welcome to my repo from Pikachu" width="512" />
  <br>
  <em>Welcome to my repo from Pikachu (generated with my LoRA checkpoints).</em>
</p>

LoRA fine-tuning of Stable Diffusion XL (SDXL) base and refiner on a curated, captioned Pokémon fanart image dataset.

- Dataset (URLs, captions, metadata):  
  Hugging Face: https://huggingface.co/datasets/Kev0208/PokeFA-pokemon-fanart-captioned  

- Dataset building pipeline (scraping → cleaning → scoring → WebDataset):  
  GitHub: https://github.com/Kev0208/PokeFA  

- PokeFA SDXL LoRA checkpoints:
  Hugging Face: {To be Updated}

---

## Repo layout

```
PokeFA-SDXL-LoRA/  
├── LICENSE                         # Project license (see file for terms)  
├── README.md                       # This document  
├── requirements.txt                # Automatically installed by the SageMaker container at job start  
├── aws/  
│   ├── sm_entry.py                 # SageMaker entry script (runs inside training container)  
│   └── submit_sm_job.py            # Client-side launcher for SageMaker jobs  
└── training/                       # All training + HPO code  
    ├── requirements.txt            # Training deps (torch, diffusers, webdataset, optuna, etc.)  
    ├── configs/                    # YAML configs (dataset/model/train/HPO/AWS)  
    │   ├── aws_sm.yaml             # SageMaker job + S3 paths + instance configs  
    │   ├── dataset.yaml            # WebDataset paths, species sampler, transforms, loader configs  
    │   ├── hpo.yaml                # HPO and eval settings  
    │   ├── model.yaml              # SDXL base/refiner, LoRA config, dtypes  
    │   └── train.yaml              # Optimization schedule, logging, checkpoints 
    ├── scripts/  
    │   ├── train.py                # CLI entry point for training (local/cluster)  
    │   └── hpo.py                  # CLI entry point for hpo (local/cluster)
    └── src/                        # Library code used by train + HPO + AWS pipelines  
        ├── dataloader.py           # WebDataset pipeline, species mixer, transforms, PyTorch loaders  
        ├── hpo.py                  # Optuna-based HPO pipeline (outer DDP + proxy evaluation)  
        ├── model.py                # SDXL assembly, LoRA injection, attention backends, dtypes  
        ├── train_loop.py           # Core training loop (DDP, Min-SNR, base vs refiner logic)  
        ├── dataloader_utils/  
        │   ├── __init__.py  
        │   ├── crops.py            # AR-aware random crop policy, deterministic center crops  
        │   ├── species_mixer.py    # Species-aware acceptance sampling  
        │   └── transfoms.py        # Transform registry, SDXL normalization, geometry fields  
        ├── hpo_utils/  
        │   ├── __init__.py  
        │   ├── metrics.py          # CLIP + aesthetic evaluation, proxy score aggregation  
        │   └── prompt_probes.py    # Prompt panel used for HPO evaluation  
        ├── model_utils/  
        │   ├── __init__.py  
        │   ├── lora.py             # LoRA target modules 
        │   ├── snr.py              # Min-SNR loss weighting helpers  
        │   ├── text.py             # SDXL text stack, cond-dropout encoding 
        │   └── utils.py            # Other model helpers  
        └── train_loop_utils/  
            ├── __init__.py  
            ├── checkpoint.py       # LoRA save/load (UNet + TE), full training state checkpointing  
            ├── dist.py             # DDP init, rank utilities, barriers, safe printing  
            ├── sched.py            # Cosine-with-warmup LR schedules  
            ├── validation.py       # Validation loop + metric aggregation across ranks
            └── utils.py            # Other train_loop helpers  
```

---

## Preparing SDXL checkpoints

- Base:  
  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0  

- Refiner:  
  https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0  

### Base

Download `stabilityai/stable-diffusion-xl-base-1.0` to `/path/to/sdxl-base`. Ensure in `model.yaml` `model.base_path` points to this directory.

The scheduler shipped there is a EulerDiscreteScheduler (intended for inference). For training, you want a DDPMScheduler. Edit `/path/to/sdxl-base/scheduler/scheduler_config.json` to:

    ```json
    {
      "_class_name": "DDPMScheduler",
      "_diffusers_version": "0.35.2",
      "beta_start": 0.00085,
      "beta_end": 0.012,
      "beta_schedule": "scaled_linear",
      "num_train_timesteps": 1000,
      "prediction_type": "epsilon",
      "clip_sample": false,
      "sample_max_value": 1.0,
      "steps_offset": 1
    }
    ```

### Refiner

Download `stabilityai/stable-diffusion-xl-refiner-1.0/unet/` to `/path/to/refiner-unet`. In `model.yaml`, `model.refiner_unet_path` should point to this directory. 

Notes:

- The refiner reuses the base `vae`, `scheduler`, and `text_encoder_2`. It does not need a full refiner pipeline. 
- In refiner stage, `text_encoder_2` is kept frozen; only the refiner `unet` LoRA trains.

---

## Preparing HPO proxy metrics

The HPO pipeline uses two CLIP-based proxies

- Prompt adherence: CLIP ViT-H/14 (LAION-2B)
  - https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K

- Aesthetic quality: CLIP ViT-L/14 + LAION-Aesthetics V2 aesthetic head
  - https://huggingface.co/openai/clip-vit-large-patch14
  - https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/sac%2Blogos%2Bava1-l14-linearMSE.pth

In `hpo.yaml`, update `clip.model_path`, `aesthetic.clip_path`, `aesthetci.head_path` with the local paths. 

---

## Preparing the dataset (WebDataset layout)

The training data is available here:

- https://huggingface.co/datasets/Kev0208/PokeFA-pokemon-fanart-captioned  

How the raw data was cleaned and converted to WebDataset is documented in:

- https://github.com/Kev0208/PokeFA  

Training expects a WebDataset layout like:

    ```
    webdataset/
    └── data/
        ├── train/
        │   ├── train-00000.tar
        │   ├── train-00001.tar
        │   └── ...
        └── val/
            └── val-00000.tar
    └── manifests/
        ├── split_stats.json
        ├── train_species.csv
        ├── val_species.csv
        ├── train_ids.txt
        └── val_ids.txt
    ```
    
### Inside each shard

Each `.tar` contains examples keyed by `image_id` and grouped as:

- Image bytes: `.jpg`, `.png`, `.webp`, etc.  
- Caption: `.txt` → hybrid caption  
- Metadata: `.json` with fields like:
    ```json
    {
      "image_id": "...",
      "split": "train" | "val",
      "source_url": "...",
      "width": 1024,
      "height": 1024,
      "ae_relevance": 98,
      "ae_aesthetic": 98,
      "species": "...",
      "species_list": [...],
      "tags": [...],
      "phash": "..."  
    }
    ```
The exact schema is what `dataloader.py` expects.

### Manifests

- `train_species.csv` : species count distribution with a `fraction` column (normalized) used by the species-aware sampler  
- `*_ids.txt`: stable lists of image IDs used to define splits  
- `split_stats.json`: summary stats, counts, and integrity hashes.

In `dataset.yaml`:

- `wds.train_urls` → `"/path/to/webdataset/data/train/train-{00000..00016}.tar"`.  
- `wds.val_urls` → `"/path/to/webdataset/data/val/val-{00000..00001}.tar"`.  
- `sampler.species_csv` → `"/path/to/webdataset/manifests/train_species.csv"`.

---

## Training configs

## Configuring training

All configs live in `training/configs/`: 

- `aws_sm.yaml` controls everything specific to SageMaker: the region, role ARN,    instance type/count, volume size, max runtime, and which DLC version to use, plus the S3 locations for code and outputs. Its `runtime` section tells `sm_entry.py` where to pull the SDXL base weights, species manifest, and WebDataset shards from in S3, and which config filenames to pass into `train.py`/`hpo.py`, as well as shard patterns and overrides for `dataset.wds.*` and `train.run_dir`.

- `dataset.yaml` controls how images and captions are read and preprocessed. It defines WebDataset shard URLs, key names for images/captions/metadata, shuffle and resampling behavior, and DataLoader settings like per-process batch size, number of workers, and pinning. It also specifies transform options (target size, crop policy, deterministic validation, normalization mode) and the species-aware sampler configuration plus its scheduling of the species-mixing coefficient over training steps.

- `model.yaml` controls how the SDXL model is assembled and which stage you are training. It selects base vs refiner (`model.stage`), points to the SDXL base snapshot and refiner UNet, configures dtypes and devices for UNet/VAEs/text encoders, and sets diffusion objective (`prediction_type`). It also contains the LoRA configs for the UNet and (for base stage only) text encoders, including ranks, alphas, dropout, target modules, master dtypes, and any refiner-specific settings like timestep band and optional TE2 PEFT merge.

- `train.yaml` controls the core training loop hyperparameters. It sets the total number of optimizer steps, warmup length, gradient accumulation (effective micro-batch size per GPU), and how often to log, validate, and checkpoint. It defines where run outputs are written (`run_dir`), how and from where to resume training, and any validation limits. It also holds the optimizer settings (AdamW betas/epsilon) and the base learning rate for LoRA parameters, which in combination with `text_encoder_lora.lr_scale` determines the effective LR for text-encoder LoRA when enabled.

- `hpo.yaml` controls the hyperparameter optimization pipeline. It specifies how many Optuna trials to run, how many training steps per trial, and which stage you are tuning (`base` or `refiner`), as well as whether to search text-encoder LoRA rank when appropriate. It also defines fixed evaluation settings (sampling steps, CFG, resolution, dtype, negative prompt), and paths for CLIP and aesthetic scoring models plus the aesthetic head. 

---

## Running training (local / cluster)

With SDXL base/refiner, dataset, and configs set up:

1. Install training deps:
    
    ```bash
    cd PokeFA-SDXL-LoRA/training
    pip install -r requirements.txt
    ```

2. Run LoRA training (example on 4 GPUs):

    ```bash
    cd PokeFA-SDXL-LoRA/training

    torchrun --nproc_per_node=4 scripts/train.py \
      --dataset configs/dataset.yaml \
      --model   configs/model.yaml \
      --train   configs/train.yaml
    ```

3. Run HPO:
    
    ```bash
    cd PokeFA-SDXL-LoRA/training

    torchrun --nproc_per_node=4 scripts/hpo.py \
      --dataset configs/dataset.yaml \
      --model   configs/model.yaml \
      --train   configs/train.yaml \
      --hpo     configs/hpo.yaml
    ```
---

## Running on AWS SageMaker

SageMaker support is split into:

- Client-side launcher: `aws/submit_sm_job.py`.
- In-container entrypoint: `aws/sm_entry.py`.

On your machine (with AWS credentials configured & training/configs/aws_sm.yaml set up): 

    ```bash
    cd PokeFA-SDXL-LoRA
    pip install "sagemaker>=2.224.0" "boto3>=1.34.0" "awscli>=2.0.0"

    python aws/submit_sm_job.py 
    ```

---

## Pipeline overview

There are five main pipelines in this repo:

1. `dataloader`  
2. `model`   
3. `train_loop`  
4. `hpo`  
5. `aws-sm`  

### 1. Dataloader (WebDataset + species mixer + geometry)

File: `training/src/dataloader.py` (+ `dataloader_utils/`)

- Builds WebDataset streams for `train` and `val` shards, optionally with:
  - `ResampledShards` for infinite training streams.
  - Sample-level shuffle buffers for good local mixing.
- Decodes `(image_bytes, caption_bytes, meta_bytes)` triplets per sample:
  - Safe alpha handling (RGBA → RGB) to avoid dark halos.
  - UTF-8 captions.
  - JSON metadata dicts.
- Applies species-aware acceptance sampling:
  - Uses `train_species.csv` and a λ-schedule from `dataset.yaml` to mix between a uniform species distribution and the empirical species distribution.
- Applies SDXL-compatible transforms:
  - AR-aware, step-dependent cropping (`random_ar_crop`) for base training.
  - Deterministic center crops (`center_crop`) for refiner/validation.
  - Normalizes images to `[-1, 1]` and attaches:
    - `original_size`, `crop_coords_top_left`, `target_size` for SDXL conditioning.
- Wraps everything in `WebLoader`, returning an iterator of `List[Dict]` batches ready for the training loop.

### 2. Model (SDXL assembly + LoRA injection)

File: `training/src/model.py` (+ `model_utils/`)

- Loads SDXL components from `model.base_path` / `model.refiner_unet_path`:
  - VAE, UNet, scheduler, text encoders, tokenizers.
- Injects LoRA adapters into:
  - UNet attention projections (`to_q`, `to_k`, `to_v`, `to_out.0`).
  - Optionally text encoder projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`) for base stage.
- Configures:
  - Attention backend (PyTorch SDPA, xformers, etc.).
  - Mixed precision dtypes (bf16/fp32) per submodule.
  - Gradient checkpointing.
- Assembles everything into an `SDXLBundle` used by the training loop:
  - `bundle.unet`, `bundle.vae`, `bundle.text_stack`, `bundle.noise_scheduler`.

### 3. Train loop (DDP, Min-SNR, base vs refiner)

File: `training/src/train_loop.py` (+ `train_loop_utils/`)

- Initializes distributed training:
  - Maps SLURM / torchrun env vars.
  - Sets CUDA devices, rank, world size.
- Builds loaders and model bundle; sets up optimizer and cosine LR scheduler with warmup.
- Handles LoRA-only optimization:
  - Collects trainable LoRA parameters (UNet + optional TE).
  - Keeps LoRA masters in fp32 (`master_dtype`), even if forwards are bf16.
  - Optionally scales TE LoRA LR by `lr_scale`.
- Implements gradient accumulation to simulate larger batches.
- Base-stage specifics:
  - Full timestep range.
  - Min-SNR-weighted loss for better high-noise learning.
  - TE LoRA updates in DDP when enabled.
- Refiner-stage specifics:
  - Restricted timestep band near low noise (`refiner_timestep_band`).
  - Plain MSE loss (no Min-SNR).
  - Text encoders frozen (using TE2 PEFT from base if configured).
- Periodically runs validation via `validation.py`, aggregating metrics across ranks.
- Saves:
  - UNet LoRA safetensors (step-tagged + rolling `last-*`).
  - TE LoRA PEFT adapters (base).
  - Full training state (optimizer, scheduler, RNG) for resume.

### 4. HPO (Optuna + CLIP/aesthetic evaluation)

File: `training/src/hpo.py` (+ `hpo_utils/`)

- Uses Optuna `TPESampler` to search over:
  - LoRA LR.
  - UNet LoRA rank (and TE LoRA rank for base).
  - Conditional dropout prob (`p_uncond`).
- For each trial:
  - Writes frozen `dataset.yaml`, `model.yaml`, `train.yaml` into `HPO_TRIALS/trial-XXXX`.
  - Launches a nested `torchrun scripts/train.py` job to perform a short proxy training run.
  - After training, evaluates the resulting LoRA on a fixed prompt panel:
    - Uses SDXL base pipeline with the trial LoRA applied.
    - Computes CLIP ViT-H/14 (LAION-2B) text–image cosine (prompt adherence).
    - Computes aesthetic scores via CLIP-ViT-L/14 + LAION-Aesthetics V2 aesthetic head.
  - Aggregates metrics across ranks and seeds, dumps `metrics.json`, and returns a scalar score to Optuna.
- After `n_trials`, writes out `best_params.yaml` capturing the best hyperparameters for a full-length training run.

### 5. AWS-SM (SageMaker integration)

Files: `aws/submit_sm_job.py`, `aws/sm_entry.py`

- `submit_sm_job.py` (client side):
  - Reads `training/configs/aws_sm.yaml`.
  - Creates a SageMaker `PyTorch` estimator with:
    - `source_dir = repo_root`.
    - `entry_point = aws/sm_entry.py`.
    - Instance type/count, S3 output path.
  - Calls `.fit()` to start the job.
- `sm_entry.py` (inside container):
  - Reads SageMaker env vars (`SM_HOSTS`, `SM_NUM_GPUS`, etc.).
  - Syncs SDXL base, WebDataset shards, species CSV from S3 to `/opt/ml/input/...`.
  - Ensures storage headroom.
  - Dispatches `torchrun training/scripts/train.py` or HPO launcher depending on `entry_mode`.
  - Writes checkpoints and outputs to `/opt/ml/model` and `/opt/ml/output` for automatic S3 upload.

---

## Outputs and checkpoints

By default, a run directory looks like:

    ```
    path/to/outputs/
    ├── cfg.yaml                                   # merged/fully-resolved training config
    ├── configs_src/                               # original source configs copied at launch
    │   ├── dataset.yaml
    │   ├── model.yaml
    │   └── train.yaml
    ├── checkpoints/
    │   ├── step-XXXXXXX-unet-lora.safetensors     # periodic UNet LoRA checkpoint 
    │   ├── ...                                    # more step-XXXXXXX-unet-lora.safetensors over time
    │   ├── last-unet-lora.safetensors             # latest UNet LoRA 
    │   ├── step-XXXXXXX-te1-peft/                 # TE1 PEFT adapter dir (base stage + TE LoRA only)
    │   ├── step-XXXXXXX-te2-peft/                 # TE2 PEFT adapter dir (base stage + TE LoRA only)
    │   ├── ...                                    # more step-XXXXXXX-te{1,2}-peft/ over time
    │   ├── last-te1-peft/                         # latest TE1 PEFT directory
    │   ├── last-te2-peft/                         # latest TE2 PEFT directory
    │   ├── state.pt                               # full training state (optimizer, scheduler, RNG, cfg)
    │   └── state_meta.json                        # lightweight metadata: {"opt_step": ..."micro_step": ...}
    ├── logs/                                      # per-rank logs                            
    └── metrics.jsonl                              # log of train/val metrics 
    ```

- UNet LoRA:  
  `last-unet-lora.safetensors` is the most convenient file to plug into diffusers at inference.

- Text encoder LoRA (base stage):  
  `*-te1-peft`, `*-te2-peft` can be merged into the original text encoders via PEFT.

---

## Acknowledgements

- Stable Diffusion XL by Stability AI.
- Diffusers and the Hugging Face ecosystem for pipelines, schedulers, and tooling.
- OpenAI and LAION for CLIP models and the aesthetic head used in HPO.
- All artists whose Pokémon fanart images are included in the PokeFA dataset (see Hugging Face dataset repo for image source urls).

If you use this repo or the resulting LoRAs in research or projects, a link back to this GitHub and the Hugging Face dataset & model repo is appreciated.
