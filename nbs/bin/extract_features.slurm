#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=26000MB
#SBATCH --cpus-per-task=1

module purge
module load singularity

while getopts ":f:c:" opt; do
  case $opt in
  f)
    featurename="$OPTARG"
    ;;
  c)
    cached="$OPTARG"
    ;;
  \?)
    echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac
done

singularity exec --nv "$NATURALCOGSCI_ROOT"/NaturalCogSci.sif \
  python3 "$NATURALCOGSCI_ROOT"/nbs/bin/extract-features-script.py \
  --featurename "$featurename" \
  --cached "$cached"

# Finish the script
exit 0
