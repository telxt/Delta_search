cd ../../get_similarity

CUDA_VISIBLE_DEVICES=2 \
python get_GraNd.py \
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
--tune_method adapter \
--quiet \
--apply_adapter \
--adapter_type houlsby \
--adapter_size 12 \
--model /data/private/lvxingtai/models/t5-base/t5.1.1.lm100k.base \
--tokenizer_path /data/private/lvxingtai/models/t5-base/t5.1.1.lm100k.base \
--source_checkpoint_path /data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing \
--output_dir '/home/lvxingtai/lxt/delta_search_code/result/adapter/GraNd' \
--task_dir "/data/private/lvxingtai/crossfit_data_for_delta/64_16n_seed42" \