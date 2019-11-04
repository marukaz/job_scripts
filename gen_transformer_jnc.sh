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
data=$1; \
fairseq-generate /gs/hs0/tga-nlp-titech/matsumaru/data/jnc/${data}_bin/ \
--path /gs/hs0/tga-nlp-titech/matsumaru/exp/jnc/$data/checkpoint_best.pt \
--gen-subset $subset \
--batch-size 64 \
--beam ${beam} > ~/home/exp/jnc/$data/${data}_gen.txt
