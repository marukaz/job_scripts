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
model=jnc_only_entail_bert_28156step; \
mkdir -p  /gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/${model}_gen; \
fairseq-generate /gs/hs0/tga-nlp-titech/matsumaru/data/only_entail_bert_28156step_jamul_ref_entail_bin/ \
--path /gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/$model/checkpoint_best.pt \
--gen-subset $subset \
--batch-size 64 \
--beam ${beam} > ~/home/entasum/fairseq_model/${model}_gen/jamul_ref_entail.out
