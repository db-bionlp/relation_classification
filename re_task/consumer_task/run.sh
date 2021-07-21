python3 main_run.py \
  --train_filename ../../dataset/consumer_complaints/train.csv  \
  --dev_filename ../../dataset/consumer_complaints/dev.csv  \
  --test_filename ../../dataset/consumer_complaints/test.csv  \
  --data_dir ../../dataset/consumer_complaints  \
  --label_file ../../dataset/consumer_complaints/label.txt  \
  --model_dir ../../saved_model/consumer_task/only_bert \
  --model only_bert \
  --model_type bert  \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=32  \
  --max_steps=-1  \
  --num_train_epochs=10 \
  --gradient_accumulation_steps=1  \
  --learning_rate=5e-5  \
  --logging_steps=700000  \
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
