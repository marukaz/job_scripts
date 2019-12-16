#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N train_jiji
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/my_dir/venvs/fairseq/bin/activate

data='jiji_head_3snt_tgt_char'
fairseq-train /gs/hs0/tga-nlp-titech/matsumaru/data/jiji/headline/${data}_bin/ \
--arch transformer_wmt_en_de \
--max-epoch 50 \
--seed 516 \
--lr 0.001 --min-lr 1e-09 \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--update-freq 16 \
--max-tokens 3584 \
--dropout 0.3 \
--clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--no-epoch-checkpoints \
--save-dir /gs/hs0/tga-nlp-titech/matsumaru/exp/jiji/fairseq/${data}
