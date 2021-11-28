#!/bin/bash -l
#
#SBATCH --job-name=COSC7502_project_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G	# memory (MB)
#SBATCH --time=0-05:00		# time (D-HH:MM)

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

D=8
N=100
M=10000

for i in {1..20}
do
    for n in {100..1000..25}
    do
        project $D $n $M >> N${n}_M${M}.txt
        echo "" >> N${n}_M${M}.txt
    done
done

for i in {1..20}
do
    for m in {12500..100000..2500}
    do
        project $D $N $m >> N${N}_M${m}.txt
        echo "" >> N${N}_M${m}.txt
    done
done

echo "Project -O3 flag tests successfully finished!"
