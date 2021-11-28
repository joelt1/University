#!/bin/bash -l
#
#SBATCH --job-name=MATH7013_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16G	# memory (GB)
#SBATCH --time=1-00:00		# time (D-HH:MM)

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "This is job '$SLURM_JOB_NAME' (id: $SLURM_JOB_ID) running on the following nodes: "
echo $SLURM_NODELIST
echo "Running with OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "Running with SLURM_TASKS_PER_NODE = $SLURM_TASKS_PER_NODE"
echo

# Activate PyTorch Python virtual environment
source ~/VirtualEnv/PyTorch/bin/activate

asset=SPXUSD
year=2017

echo "Running fdrnn.py on ${asset} ${year} dataset..."

# Train and test FDRNN model on selected dataset
time python ../fdrnn.py $asset $year >> ${asset}_${year}.txt
echo

echo "FDRNN training and testing on ${asset} ${year} dataset successfully finished!"
