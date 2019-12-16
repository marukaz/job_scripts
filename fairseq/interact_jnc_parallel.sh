for c in {a..p}; do
  qsub -p -4 -g tga-nlp-titech interact_transformer_jnc.sh gingo_filtered_bert ~/my_dir/data/jnc/gingo_unlabeled_bert/xa$c
done
