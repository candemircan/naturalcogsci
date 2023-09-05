#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=26000MB
#SBATCH  --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60G

module purge
module load anaconda/3/2023.03

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

python "$NATURALCOGSCI_ROOT"/nbs/bin/extract-features-script.py \
  --featurename "$featurename" \
  --cached "$cached"

# Finish the script
exit 0
