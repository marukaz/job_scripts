#!/bin/sh
## current working directory
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=00:20:00
#$ -N gen_lstm
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.gen_lstm
#$ -e e.gen_lstm

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/venvs/fairseq/bin/activate

dataset="headline_5snt"; \
beam=5; subset="test"; \
fairseq-generate /gs/hs0/tga-nlp-titech/matsumaru/data/jiji/merged_filtered/fairseq_dataset/${dataset}_bin/ \
--path /gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/jiji_lstm_${dataset}/checkpoint_best.pt \
--gen-subset $subset \
--batch-size $beam \
--beam ${beam} > ~/home/entasum/fairseq_model/jiji_gen/lstm_beam${beam}_${dataset}_${subset}.json
