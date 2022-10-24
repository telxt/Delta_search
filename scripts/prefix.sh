cd ..

TASKS="all_task"
DATA_DIR="/data/private/lvxingtai/crossfit_data_for_delta/64_16n_seed42"
TUNE_METHOD=prefix
IDENTIFIER=full_data_prefix
SEED=42
PREFIX_R=24
PREFIX_NUM=120

echo "Task: $TASK, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=0 \
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python tune_hps_singletask.py \
--do_train \
--do_predict \
--learning_rate_list 5e-4 \
--bsz_list 8 \
--train_iters 100000 \
--warmup_steps 200 \
--valid_interval 100 \
--log_interval 100 \
--early_stop 10 \
--predict_batch_size 80 \
--one_prefix \
--quiet \
--apply_prefix \
--prefix_r ${PREFIX_R} \
--prefix_num ${PREFIX_NUM} \
--tune_method ${TUNE_METHOD} \
--seed ${SEED} \
--model /data/private/lvxingtai/models/t5-base/t5.1.1.lm100k.base \
--tokenizer_path /data/private/lvxingtai/models/t5-base/t5.1.1.lm100k.base \
--output_dir '/home/lvxingtai/lxt/delta_search_code/result/prefix/random/lr_5e-4_bsz_8_vi_100_er_10' \
--task_dir ${DATA_DIR} \