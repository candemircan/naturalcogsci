#!/bin/bash

#SBATCH -J linear_models
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate td_llama

while getopts ":e:f:t:r:" opt; do
  case $opt in
  e)
    experiment="$OPTARG"
    ;;
  f)
    feature="$OPTARG"
    ;;
  t)
    transform="$OPTARG"
    ;;
  r)
    regularisation="$OPTARG"
    ;;
  \?)
    echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac
done

python3 "$NATURALCOGSCI_ROOT"/bin/run_learners.py \
  --experiment "$experiment" \
  --features "$feature" \
  --transform "$transform" \
  --regularisation "$regularisation"

# Finish the script
exit 0
