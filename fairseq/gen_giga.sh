#!/bin/sh
## current working directory
#$ -cwd
#$ -l h_rt=01:00:00
#$ -N gen_giga
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

if [ `whoami` = 'acb11164rn' ]; then
  module load python
fi

module load cuda/10.1
module load cudnn/7.6
module load nccl/2.4

source ~/my_dir/venvs/fairseq/bin/activate

beam=5; subset="test" \
model=$1 \
data=$1 \
# mkdir -p /gs/hs0/tga-nlp-titech/matsumaru/exp/giga/$data \
fairseq-generate ~/my_dir/data/giga/${data}_bin \
--path ~/my_dir/exp/giga/$model/checkpoint_best.pt \
--gen-subset $subset \
--batch-size 128 \
--beam ${beam} > ~/my_dir/exp/giga/$model/gen.txt
