#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=01:00:00
#$ -N interact
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/venvs/fairseq/bin/activate

beam=5; \
model=jiji_head_3snt; \
input=$1; \
dirpath=${input%/*}; \
dir=${dirpath##*/}; \
mkdir -p /gs/hs0/tga-nlp-titech/matsumaru/exp/jiji/fairseq/$dir
fairseq-interactive  /gs/hs0/tga-nlp-titech/matsumaru/data/jiji/headline/${model}_bin --input $input \
--path /gs/hs0/tga-nlp-titech/matsumaru/exp/jiji/fairseq/$model/checkpoint_best.pt \
--buffer-size 64 \
--batch-size 64 \
--beam ${beam} > /gs/hs0/tga-nlp-titech/matsumaru/exp/jiji/fairseq/$dir/${input##*/}_gen.txt
