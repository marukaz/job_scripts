#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=06:00:00
#$ -N gen
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/venvs/fairseq/bin/activate

beam=5; subset="test"; \
model=$1; \
data=$1; \
# mkdir -p /gs/hs0/tga-nlp-titech/matsumaru/exp/giga/$data; \
fairseq-generate /gs/hs0/tga-nlp-titech/matsumaru/data/giga/${data}_bin \
--path /gs/hs0/tga-nlp-titech/matsumaru/exp/giga/$model/checkpoint_best.pt \
--gen-subset $subset \
--batch-size 128 \
--beam ${beam} > /gs/hs0/tga-nlp-titech/matsumaru/exp/giga/$model/gen.txt
