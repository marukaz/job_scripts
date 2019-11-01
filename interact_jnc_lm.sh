#!/bin/sh
## current working directory
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=01:00:00
#$ -N gen_jnc_lm
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/venvs/fairseq/bin/activate

beam=5; subset="test"; \
data=$1; \
model=${data}_lm; \
prefix=$2; \
# mkdir -p /gs/hs0/tga-nlp-titech/matsumaru/exp/giga/$data; \
fairseq-interactive /gs/hs0/tga-nlp-titech/matsumaru/data/${data}_bin \
--input /gs/hs0/tga-nlp-titech/matsumaru/data/jnc_fairseq_3snt/test.tgt \
--task language_modeling \
--path /gs/hs0/tga-nlp-titech/matsumaru/exp/jnc/$model/checkpoint_best.pt \
--buffer-size 128 \
--prefix-size $prefix \
--beam ${beam} > /gs/hs0/tga-nlp-titech/matsumaru/exp/jnc/$model/gen_pre${prefix}.txt
