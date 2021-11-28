#!/bin/bash -l
#
#SBATCH --job-name=COSC7502_project_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G	# memory (MB)
#SBATCH --time=0-10:00		# time (D-HH:MM)

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
    echo "You probably need to compile project.cu first"
    exit 2
fi

D=8
N=100
M=10000

for i in {1..5}
do
    for NN in {100..1000..25}
    do
        project $D $NN $M >> N${NN}_M${M}.txt
        echo "" >> N${NN}_M${M}.txt
        sleep 1s
    done
done

for i in {1..5}
do
    for MM in {12500..100000..2500}
    do
        project $D $N $MM >> N${N}_M${MM}.txt
        echo "" >> N${N}_M${MM}.txt
        sleep 1s
    done
done

echo "Project verification tests successfully finished!"
