#!/bin/sh
## current working directory
#$ -cwd
#$ -l h_rt=48:00:00
#$ -N train_giga
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh


if [ `whoami` = 'acb11164rn' ]; then
  module load python
fi

module load cuda/10.1
module load cudnn/7.6
module load nccl/2.4

source ~/my_dir/venvs/fairseq/bin/activate
data=$1;
fairseq-train ~/my_dir/data/giga/${data}_bin/ \
--arch transformer_wmt_en_de \
--seed 516 \
--max-epoch 50 \
--lr 0.0005 --min-lr 1e-09 \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--update-freq 4 \
--max-tokens 3584 \
--clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
--encoder-normalize-before  --decoder-normalize-before \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--no-epoch-checkpoints \
--save-dir ~/my_dir/exp/giga/${data}
