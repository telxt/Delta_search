TASKS="all_task"
DATA_DIR="/data/private/lvxingtai/crossfit_data_for_delta/256_64n_seed42"
TUNE_METHOD=adapter
ADAPTER_SIZE=12
IDENTIFIER=full_data_adapter

echo "Task: $TASK, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=0 \
python get_KL_divergence.py \
--do_train \
--do_predict \
--learning_rate_list 5e-4 \
--bsz_list 8 \
--train_iters 100000 \
--warmup_steps 200 \
--valid_interval 100 \
--log_interval 100 \
--early_stop 10 \
--predict_batch_size 200 \
--tune_method ${TUNE_METHOD} \
--quiet \
--apply_adapter \
--adapter_type houlsby \
--adapter_size ${ADAPTER_SIZE} \
--model /data/private/lvxingtai/models/t5-base/t5.1.1.lm100k.base \
--tokenizer_path /data/private/lvxingtai/models/t5-base/t5.1.1.lm100k.base \
--output_dir '/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks_256shot/KL_divergence' \
--task_dir ${DATA_DIR} \