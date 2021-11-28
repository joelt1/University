#!/bin/bash -l
#
#SBATCH --job-name=COSC7502_a2_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G	# memory (MB)
#SBATCH --time=0-10:00		# time (D-HH:MM)

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "Running with OMP_NUM_THREADS = " $OMP_NUM_THREADS
echo "Running with MKL_NUM_THREADS = " $MKL_NUM_THREADS
echo "This is job '$SLURM_JOB_NAME' (id: $SLURM_JOB_ID) running on the following nodes: "
echo $SLURM_NODELIST
echo "Running with OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "Running with SLURM_TASKS_PER_NODE = $SLURM_TASKS_PER_NODE"
echo

if [ ! -f ../../a2_serial ]
then
    echo "Unable to find Serial implementation code"
    echo "You probably need to compile a2_serial.cpp first"
    exit 2
fi

SECONDS=0
for N in {1000..10000..1000}
do
    for i in {1..10}
    do
        echo "N = ${N}, run ${i}" >> serial_results.txt
        ../../a2_serial $N >> serial_results.txt
        echo "" >> serial_results.txt
    done
done

echo "Serial implementation test finished!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
