#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N jnc_train
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/my_dir/venvs/fairseq/bin/activate

data=$1
fairseq-train /gs/hs0/tga-nlp-titech/matsumaru/data/jnc/${data}_bin/ \
--arch transformer_wmt_en_de \
--seed 516 \
--max-epoch 100 \
--lr 0.001 --min-lr 1e-09 \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--max-tokens 3584 \
--dropout 0.3 \
--clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--no-epoch-checkpoints \
--save-dir /gs/hs0/tga-nlp-titech/matsumaru/exp/jnc/${data}_official_param
