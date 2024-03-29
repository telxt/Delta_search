cd ..

CUDA_VISIBLE_DEVICES=4 \
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
--tune_method adapter \
--quiet \
--apply_adapter \
--adapter_type houlsby \
--adapter_size 12 \
--model /data/private/lvxingtai/models/t5-base/t5.1.1.lm100k.base \
--tokenizer_path /data/private/lvxingtai/models/t5-base/t5.1.1.lm100k.base \
--output_dir '/home/lvxingtai/lxt/delta_search_code/result/adapter/final_loss/lr_5e-4_bsz_8_vi_100_er_10' \
--task_dir "/data/private/lvxingtai/crossfit_data_for_delta/64_16n_seed42"