for split in {k..y}; do
  qsub -l $G4=1 -g $GR gen_unilm.sh filtered_roberta_wp  10 xa${split}
done
