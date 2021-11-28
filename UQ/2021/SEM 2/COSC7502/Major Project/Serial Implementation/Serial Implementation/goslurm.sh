#!/bin/bash -l
#
#SBATCH --job-name=COSC7502_project_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G	# memory (MB)
#SBATCH --time=0-00:30		# time (D-HH:MM)

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "This is job '$SLURM_JOB_NAME' (id: $SLURM_JOB_ID) running on the following nodes: "
echo $SLURM_NODELIST
echo "Running with OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "Running with SLURM_TASKS_PER_NODE = $SLURM_TASKS_PER_NODE"
echo

if [ ! -f project ]
then
    echo "Unable to find project"
    echo "You probably need to compile project.cpp first"
    exit 2
fi

echo "Runtime for program:"
time ./project 8 1000 10000 >> project_output.txt
