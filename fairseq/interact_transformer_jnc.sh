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
model=jnc_fairseq_3snt; \
input=$1; \
dirpath=${input%/*}; \
dir=${dirpath##*/}; \
mkdir -p /gs/hs0/tga-nlp-titech/matsumaru/exp/jnc/$dir
fairseq-interactive  /gs/hs0/tga-nlp-titech/matsumaru/data/${model}_bin --input $input \
--path /gs/hs0/tga-nlp-titech/matsumaru/exp/jnc/$model/checkpoint_best.pt \
--buffer-size 1 \
--batch-size 1 \
--beam ${beam} > /gs/hs0/tga-nlp-titech/matsumaru/exp/jnc/$dir/${input##*/}_gen.txt
