#!/bin/sh
## current working directory
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=01:00:00
#$ -N gen
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/venvs/fairseq/bin/activate

beam=5; subset="test"; \
model=jiji_head_3snt; \
data=$1; \
mkdir -p /gs/hs0/tga-nlp-titech/matsumaru/exp/jiji/fairseq/$data; \
fairseq-generate /gs/hs0/tga-nlp-titech/matsumaru/data/jiji/headline/${data}_bin \
--path /gs/hs0/tga-nlp-titech/matsumaru/exp/jiji/fairseq/$model/checkpoint_best.pt \
--gen-subset $subset \
--batch-size 64 \
--beam ${beam} > /gs/hs0/tga-nlp-titech/matsumaru/exp/jiji/fairseq/$data/${model}_gen.txt
