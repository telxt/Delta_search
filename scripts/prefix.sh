cd ..

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
--prefix_r 24 \
--prefix_num 120 \
--tune_method prefix \
--seed 42 \
--model /data/private/lvxingtai/models/t5-base/t5.1.1.lm100k.base \
--tokenizer_path /data/private/lvxingtai/models/t5-base/t5.1.1.lm100k.base \
--output_dir '/home/lvxingtai/lxt/delta_search_code/result/prefix/random/lr_5e-4_bsz_8_vi_100_er_10' \
--task_dir "/data/private/lvxingtai/crossfit_data_for_delta/64_16n_seed42" \