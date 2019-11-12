#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N lc_transformer
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3
module load nccl

source ~/venvs/fairseq/bin/activate
export PYTHONPATH=/gs/hs0/tga-nlp-titech/matsumaru/repos/lab-api-server

data='gigaword_3snt_tgt_char'
fairseq-train /gs/hs0/tga-nlp-titech/matsumaru/data/giga/${data}_bin/ \
--user-dir ~/home/repos/lab-api-server/length_control \
--arch lc_transformer \
--task lc_translation \
--represent-length-by-lrpe \
--ordinary-sinpos \
--seed 2723 \
--share-all-embeddings \
--max-epoch 100 \
--lr 0.0005 --min-lr 1e-09 \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--update-freq 16 \
--max-tokens 4000 \
--dropout 0.3 \
--clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--no-epoch-checkpoints \
--save-dir /gs/hs0/tga-nlp-titech/matsumaru/exp/giga/LRPE_PE_${data}
