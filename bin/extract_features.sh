#!/bin/bash

models=$(jq -c -r 'keys_unsorted[]' "$NATURALCOGSCI_ROOT"/data/model_configs.json)

for model in $models; do
    sbatch "$NATURALCOGSCI_ROOT"/bin/extract_features.slurm \
        -f "$model" \
        -c true
done
