#!/bin/bash

models=(
    dinov2-vit-small-p14
    dinov2-vit-base-p14
    dinov2-vit-large-p14
    dinov2-vit-giant-p14
    DreamSim_open_clip_vitb32
    DreamSim_clip_vitb32
    OpenCLIP_ViT-B-32_laion400m_e32
    OpenCLIP_ViT-L-14_laion400m_e32
    OpenCLIP_ViT-H-14_laion2b_s32b_b79k
    OpenCLIP_ViT-g-14_laion2b_s12b_b42k
    Resnet50_ecoset
    Inception_ecoset
    slip_slip_small
    clip_slip_small
    simclr_slip_small
    slip_slip_base
    clip_slip_base
    simclr_slip_base
    slip_slip_large
    clip_slip_large
    simclr_slip_large

)

for model in "${models[@]}"; do
    sbatch "$NATURALCOGSCI_ROOT"/nbs/bin/extract_features_v2.slurm \
        -f "$model" \
        -c true
done
