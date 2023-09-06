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
)

for model in "${models[@]}"; do
    sbatch "$NATURALCOGSCI_ROOT"/nbs/bin/extract_features_v2.slurm \
        -f "$model" \
        -c true
done
