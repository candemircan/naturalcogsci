#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=5-00:00:00
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=20000MB
#SBATCH --cpus-per-task=1

module purge
module load singularity

while getopts ":f:t:" opt; do
  case $opt in
  f)
    features="$OPTARG"
    ;;
  t)
    target="$OPTARG"
    ;;
  \?)
    echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac
done

singularity exec "$NATURALCOGSCI_ROOT"/NaturalCogSci.sif \
  python3 "$NATURALCOGSCI_ROOT"/nbs/bin/cka-script.py \
  -f "$features" -t "$target"

# Finish the script
exit 0
