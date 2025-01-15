#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=26000MB
#SBATCH --cpus-per-task=1

module purge
module load virtualenv

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

python3 "$NATURALCOGSCI_ROOT"/bin/extract_features.py \
  --featurename "$featurename" \
  --cached "$cached"

# Finish the script
exit 0
