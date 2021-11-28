#!/bin/bash -l
#
#SBATCH --job-name=COSC7502_a2_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G	# memory (MB)
#SBATCH --time=0-00:10		# time (D-HH:MM)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "Running with OMP_NUM_THREADS = " $OMP_NUM_THREADS
echo "Running with MKL_NUM_THREADS = " $MKL_NUM_THREADS
echo "This is job '$SLURM_JOB_NAME' (id: $SLURM_JOB_ID) running on the following nodes: "
echo $SLURM_NODELIST
echo "Running with OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "Running with SLURM_TASKS_PER_NODE = $SLURM_TASKS_PER_NODE"
echo

file=a2_cuda
if [ ! -f ./$file ]
then
    echo "Unable to find the compiled code"
    echo "You probably need to compile the .cpp or .cu file first"
    exit 2
fi

SECONDS=0
N=10000
time ./$file $N >> results.txt
echo "" >> results.txt
echo

echo "Implementation test finished!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
