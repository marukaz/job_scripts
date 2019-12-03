#!/bin/sh
## current working directory
#$ -cwd
#$ -l h_rt=72:00:00
#$ -N gen_unilm
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/10.1
module load cudnn/7.6
module load nccl/2.4
export PATH=/apps/gcc/7.3.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/gcc/7.3.0/lib64:$LD_LIBRARY_PATH
export PYTHONPATH='/home/acb11164rn/.local/lib/python3.6/site-packages/'

if [ `whoami` = 'acb11164rn' ]; then
  module load python
fi

cd ~/my_dir/repos/unilm
source venv/bin/activate
export PYTHONPATH="$PYTHONPATH:/fs1/groups1/gcb50243/matsumaru/repos/unilm/src"
cd src

# run decoding
MODEL_DIR=$1
MODEL_NUM=$2
DATA_DIR=~/my_dir/data/giga/filtered_roberta_wp
MODEL_RECOVER_PATH=~/my_dir/exp/unilm/filtered_roberta_wp/bert_save/model.8.bin
export PYTORCH_PRETRAINED_BERT_CACHE=~/my_dir/exp/unilm/filtered_roberta_wp/bert-cased-pretrained-cache
EVAL_SPLIT=test
python biunilm/decode_seq2seq.py --fp16 --amp --bert_model bert-large-cased --new_segment_ids --mode s2s --need_score_traces \
  --input_file ${DATA_DIR}/${EVAL_SPLIT}.src --split ${EVAL_SPLIT} --tokenized_input \
  --model_recover_path ~/my_dir/exp/unilm/$MODEL_DIR/bert_save/model.${MODEL_NUM}.bin \
  --max_seq_length 192 --max_tgt_length 32 \
  --batch_size 64 --beam_size 5 --length_penalty 0 \
  --forbid_duplicate_ngrams --forbid_ignore_word "."
