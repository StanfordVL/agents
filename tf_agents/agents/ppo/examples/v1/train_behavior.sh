gpu_c="1"
gpu_g="0"

if [[ -z "${CONFIG_FILE}" ]]; then
  config_file="examples/configs/behavior_reshelf_book.yaml"
else
  config_file="${CONFIG_FILE}"
fi

if [[ -z "${LOG_DIR}" ]]; then
  log_dir="test"
else
  log_dir="${LOG_DIR}"
fi

echo "config_file:" $config_file
echo "log_dir:" $log_dir

python -u train_eval.py \
    --root_dir $log_dir \
    --config_file $config_file \
    --replay_buffer_capacity 10000 \
    --num_eval_episodes 2 \
    --eval_interval 10000000 \
    --gpu_c $gpu_c \
    --gpu_g $gpu_g \
    --num_parallel_environments 6 \
    --model_ids Rs_int,Rs_int,Rs_int,Rs_int,Rs_int,Rs_int \
    --action_filter mobile_manipulation
