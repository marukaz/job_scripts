#!/bin/sh
## current working directory
#$ -cwd
#$ -l h_rt=01:00:00
#$ -N interact_giga
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh


source ~/my_dir/venvs/fairseq/bin/activate

beam=5; subset="test"; \
DATA=$1; \
# mkdir -p /gs/hs0/tga-nlp-titech/matsumaru/exp/giga/$data; \
fairseq-interactive ~/my_dir/data/giga/${DATA}_bin \
--path ~/my_dir/exp/giga/$DATA/checkpoint_best.pt \
--beam $beam \
--remove-bpe " ##" --buffer-size 64 \
--input ~/my_dir/data/giga/giga_1snt_small_test_wp/test.src > ~/my_dir/exp/giga/$DATA/gen_test10k.txt
