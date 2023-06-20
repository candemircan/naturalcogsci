#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=20000MB
#SBATCH --cpus-per-task=1

module purge
module load singularity

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

singularity exec "$NATURALCOGSCI_ROOT"/NaturalCogSci.sif \
  python3 "$NATURALCOGSCI_ROOT"/nbs/bin/run-learners-script.py \
  --experiment "$experiment" \
  --features "$feature" \
  --transform "$transform" \
  --regularisation "$regularisation"

# Finish the script
exit 0
