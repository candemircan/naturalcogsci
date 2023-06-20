#!/bin/bash

embeddings=$(jq -c -r 'keys_unsorted[]' "$NATURALCOGSCI_ROOT"/data/model_configs.json)

for experiment in reward_learning category_learning; do
    for features in $embeddings; do
        for transform in original pca; do
            for regularisation in l1 l2; do

                sbatch "$NATURALCOGSCI_ROOT"/nbs/bin/run_learners.slurm \
                    -e $experiment -f "$features" -t $transform -r $regularisation
            done
        done
    done
done
