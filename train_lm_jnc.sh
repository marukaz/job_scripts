#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N train_lm
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/venvs/fairseq/bin/activate

data='jnc_fairseq_3snt_only_target'
fairseq-train --task language_modeling /gs/hs0/tga-nlp-titech/matsumaru/data/${data}_bin/ \
--arch transformer_lm \
--max-epoch 50 \
--lr 0.0005 --min-lr 1e-09 \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--update-freq 16 \
--max-tokens 3584 \
--clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--no-epoch-checkpoints \
--save-dir /gs/hs0/tga-nlp-titech/matsumaru/exp/jnc/${data}_lm
