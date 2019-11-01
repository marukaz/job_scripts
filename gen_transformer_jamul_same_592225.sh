#!/bin/sh
## current working directory
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=01:00:00
#$ -N gen
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.gen
#$ -e e.gen

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/venvs/fairseq/bin/activate

beam=5; subset="test"; \
model=jnc_same_as_659602; \
mkdir -p  /gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/${model}_gen; \
fairseq-generate /gs/hs0/tga-nlp-titech/matsumaru/data/same_as_only_entail_659602_jamul_test_bin/ \
--path /gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/$model/checkpoint_best.pt \
--gen-subset $subset \
--batch-size 128 \
--beam ${beam} > ~/home/entasum/fairseq_model/${model}_gen/jamul_test.out
