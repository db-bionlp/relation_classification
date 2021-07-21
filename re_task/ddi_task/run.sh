python3 main_run.py \
  --train_filename ../../dataset/DDI_corpus/train.tsv  \
  --dev_filename ../../dataset//DDI_corpus/test.tsv  \
  --test_filename ../../dataset//DDI_corpus/test.tsv  \
  --data_dir ../../dataset/DDI_corpus  \
  --label_file ../../dataset/DDI_corpus/label.csv  \
  --model_dir ../../saved_model/ddi_task/only_bert \
  --model only_bert \
  --model_type bert  \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=32  \
  --max_steps=-1  \
  --num_train_epochs=3 \
  --gradient_accumulation_steps=1  \
  --learning_rate=5e-5  \
  --logging_steps=25000000000000  \
  --save_steps=100 \
  --adam_epsilon=1e-8  \
  --warmup_steps=0  \
  --dropout_rate=0.1  \
  --weight_decay=0.0  \
  --seed=42  \
  --max_grad_norm=1.0  \
  --do_train \
#  --do_eval \
#  --do_test \

python3 main_run.py \
  --train_filename ../../dataset/DDI_corpus/train.tsv  \
  --dev_filename ../../dataset//DDI_corpus/test.tsv  \
  --test_filename ../../dataset//DDI_corpus/test.tsv  \
  --data_dir ../../dataset/DDI_corpus  \
  --label_file ../../dataset/DDI_corpus/label.csv  \
  --model_dir ../../saved_model/ddi_task/bert_center \
  --model bert_center \
  --model_type bert  \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=32  \
  --max_steps=-1  \
  --num_train_epochs=3 \
  --gradient_accumulation_steps=1  \
  --learning_rate=5e-5  \
  --logging_steps=25000000000000  \
  --save_steps=100 \
  --adam_epsilon=1e-8  \
  --warmup_steps=0  \
  --dropout_rate=0.1  \
  --weight_decay=0.0  \
  --seed=42  \
  --max_grad_norm=1.0  \
  --do_train \
#  --do_eval \
#  --do_test \