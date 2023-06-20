#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --partition compute
#SBATCH --mem-per-cpu=10000MB

module purge
module load singularity

while getopts ":e:f:t:r:" opt; do
  case $opt in
  e)
    experiment="$OPTARG"
    ;;
  f)
    features="$OPTARG"
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

singularity exec "$NATURALCOGSCI_ROOT"/NaturalCogSci.sif \
  Rscript "$NATURALCOGSCI_ROOT"/nbs/bin/loo-cv-script.R \
  "$features" "$experiment" "$regularisation" "$transform"

# Finish the script
exit 0
