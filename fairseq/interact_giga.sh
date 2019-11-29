#!/bin/sh
## current working directory
#$ -cwd
#$ -l h_rt=24:00:00
#$ -N interact_giga
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/10.1
module load cudnn/7.6
module load nccl
module load python

source ~/my_dir/venvs/fairseq/bin/activate

beam=5 \
MODEL=$1 \
INPUT=$2 \
OUT=$3 \
# mkdir -p /gs/hs0/tga-nlp-titech/matsumaru/exp/giga/$data; \
fairseq-interactive ~/my_dir/data/giga/${MODEL}_bin \
--path ~/my_dir/exp/giga/$MODEL/checkpoint_best.pt \
--beam $beam \
--remove-bpe " ##" --buffer-size 64 \
--input $INPUT > ~/my_dir/exp/giga/$MODEL/$OUT
