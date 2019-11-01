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
model=$1; \
input=$2; \
mkdir -p /gs/hs0/tga-nlp-titech/matsumaru/exp/adversarial/$model; \
fairseq-interactive  /gs/hs0/tga-nlp-titech/matsumaru/data/jiji/headline/${model}_bin --input /gs/hs0/tga-nlp-titech/matsumaru/adversarial/$data \
--path /gs/hs0/tga-nlp-titech/matsumaru/exp/$model/checkpoint_best.pt \
--buffer-size 64 \
--batch-size 64 \
--beam ${beam} > /gs/hs0/tga-nlp-titech/matsumaru/exp/adversarial/$model/${input}_gen.txt
