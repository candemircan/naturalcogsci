#!/bin/bash

embeddings=$(jq -c -r 'keys_unsorted[]' "$NATURALCOGSCI_ROOT"/data/model_configs.json)

for features in $embeddings; do
    uv run "$NATURALCOGSCI_ROOT"/bin/r2_class_sep.py -f "$features"
done
