#!/bin/sh
## current working directory
#$ -cwd
#$ -l h_rt=24:00:00
#$ -N train_cnndm
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/10.1
module load cudnn/7.6
module load nccl

if [ `whoami` = 'acb11164rn' ]; then
  module load python
fi

source ~/my_dir/venvs/fairseq/bin/activate
data=$1;
fairseq-train ~/groupdisk/data/${data}_bin/ \
--arch transformer \
--seed 516 \
--max-epoch 100 \
--lr 0.0005 --min-lr 1e-09 \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--update-freq 4 \
--max-tokens 3584 \
--dropout 0.3 \
--clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--save-interval 5 \
--save-dir ~/my_dir/exp/cnndm/$data
