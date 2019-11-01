#!/bin/sh
## current working directory
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=00:20:00
#$ -N beam
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.beam
#$ -e e.beam

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/venvs/fairseq/bin/activate

dataset="summary_1snt"; \
beam=5; subset="test"; \
fairseq-generate /gs/hs0/tga-nlp-titech/matsumaru/data/jiji/merged_filtered/fairseq_dataset/${dataset}_bin/ \
--path /gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/jiji_${dataset}/checkpoint_best.pt \
--gen-subset $subset \
--batch-size $beam \
--beam ${beam} > ~/home/entasum/fairseq_model/jiji_gen/beam${beam}_${dataset}_${subset}.json
