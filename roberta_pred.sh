#!/bin/sh
## current working directory
#$ -cwd
#$ -l h_rt=72:00:00
#$ -N roberta_pred
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh


cd ~/my_dir/repos/roberta_giga_ent
source venv/bin/activate

module load cuda/10.1
module load cudnn/7.6
module load nccl
module load python

# run fine-tuning
export CUDA_VISIBLE_DEVICES=0,1,2,3
python predict.py --model-dir checkpoints/roberta.large.mnli --data-dir ~/my_dir/data/giga/giga_1snt_1000_test_rep_no_wp
