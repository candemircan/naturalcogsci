#!/bin/bash

for target in task clip_RN50 clip_RN101 clip_RN50x4 clip_ViT-B_16 clip_ViT-B_32 clip_ViT-L_14; do
    sbatch "$NATURALCOGSCI_ROOT"/nbs/bin/cka.slurm \
        -f all -t $target
done
