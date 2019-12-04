for c in {a..y}; do
  qsub -l $G4=1 -g $GR interact_giga.sh filtered_roberta_wp ~/my_dir/data/giga/unlabeled_from_roberta/xa$c unlabeled_xa${c}.txt
done
