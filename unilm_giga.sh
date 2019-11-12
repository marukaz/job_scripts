#!/bin/sh
## current working directory
#$ -cwd
#$ -l h_rt=72:00:00
#$ -N unilm_giga
#$ -m abe
#$ -M kopamaru@gmail.com

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/10.0
module load cudnn/7.5
module load nccl
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

# run fine-tuning
DATA_DIR=~/my_dir/data/giga/giga_1snt_1000_test_rep
OUTPUT_DIR=~/my_dir/exp/unilm/giga_1snt_1000_test_rep
MODEL_RECOVER_PATH=~/my_dir/exp/unilm/unilmv1-large-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=~/my_dir/exp/unilm/giga_1snt_1000_test_re/pbert-cased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
python biunilm/run_seq2seq.py --do_train --fp16 --amp --num_workers 0 \
  --bert_model bert-large-cased --new_segment_ids --tokenized_input \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 192 --max_position_embeddings 192 \
  --trunc_seg a --always_truncate_tail --max_len_a 0 --max_len_b 64 \
  --mask_prob 0.7 --max_pred 48 \
  --train_batch_size 64 --gradient_accumulation_steps 1 \
  --learning_rate 0.00003 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 10
