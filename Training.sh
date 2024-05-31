#! /bin/bash
######## Part 1 #########
# Script parameters     #
#########################
  
# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu
  
# Specify the QOS, mandatory option
#SBATCH --qos=normal
  
# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
#SBATCH --account=junogpu
  
# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=pi+
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=log_train.out
#SBATCH --error=log_train.err
  
# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --cpus-per-task=1  
#SBATCH --mem-per-cpu=40000
#
# Specify how many GPU cards to us:
#SBATCH --gres=gpu:v100:1
######## Part 2 ######
# Script workload    #
######################
  
# Replace the following lines with your real workload
  
# list the allocated hosts
workpath=pwd
### --training ##
python dedx_flow.py --batch_size 128 --num_epochs 100 --training --num_block=3 --workpath $workpath -p=pi+  --data_dir=$workpath/dataset/ --output_dir=$workpath/results_pi+/ --results_file=train.txt --save_model_name='pi+.pth'
python dedx_flow.py --batch_size 128 --num_epochs 100 --training --num_block=3 --workpath $workpath -p=pi-  --data_dir=$workpath/dataset/ --output_dir=$workpath/results_pi-/ --results_file=train.txt --save_model_name='pi-.pth'
python dedx_flow.py --batch_size 128 --num_epochs 100 --training --num_block=3 --workpath $workpath -p=k-   --data_dir=$workpath/dataset/ --output_dir=$workpath/results_k-/  --results_file=train.txt --save_model_name='k-.pth'
python dedx_flow.py --batch_size 128 --num_epochs 100 --training --num_block=3 --workpath $workpath -p=k+   --data_dir=$workpath/dataset/ --output_dir=$workpath/results_k+/  --results_file=train.txt --save_model_name='k+.pth'
python dedx_flow.py --batch_size 128 --num_epochs 100 --training --num_block=3 --workpath $workpath -p=p-   --data_dir=$workpath/dataset/ --output_dir=$workpath/results_p-/  --results_file=train.txt --save_model_name='p-.pth'
python dedx_flow.py --batch_size 128 --num_epochs 100 --training --num_block=3 --workpath $workpath -p=p+   --data_dir=$workpath/dataset/ --output_dir=$workpath/results_p+/  --results_file=train.txt --save_model_name='p+.pth'
