TASKS="all_task"
DATA_DIR="/data/private/lvxingtai/crossfit_data_for_delta/64_16n_seed42"
TUNE_METHOD=adapter
ADAPTER_SIZE=12
IDENTIFIER=full_data_adapter

echo "Task: $TASK, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=3 \
python get_block.py \
--do_train \
--do_predict \
--learning_rate_list 5e-4 \
--bsz_list 8 \
--train_iters 100000 \
--warmup_steps 200 \
--valid_interval 100 \
--log_interval 100 \
--early_stop 10 \
--predict_batch_size 50 \
--tune_method ${TUNE_METHOD} \
--quiet \
--apply_adapter \
--adapter_type houlsby \
--adapter_size ${ADAPTER_SIZE} \
--model pretrained_models/t5.1.1.lm100k.base \
--tokenizer_path pretrained_models/t5.1.1.lm100k.base \
--output_dir '/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block' \
--task_dir ${DATA_DIR} \