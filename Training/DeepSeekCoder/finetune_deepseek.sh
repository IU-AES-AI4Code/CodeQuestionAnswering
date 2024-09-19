DATA_PATH="/home/user/georgy/CodeQAArticle/DeepSeekCoder/train_grammar_corrected.json"
OUTPUT_PATH="/home/user/georgy/CodeQAArticle/DeepSeekCoder/DeepSeekGrammarCorrected"
MODEL="deepseek-ai/deepseek-coder-6.7b-instruct"

cd DeepSeek-Coder/finetune && CUDA_VISIBLE_DEVICES=1 python3 finetune_deepseekcoder.py \
    --model_name_or_path $MODEL \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    # --max_steps 10001 \
    # --model_max_length 1024 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1000 \
    --lr_scheduler_type "cosine" \
    # --gradient_checkpointing True \
    # --report_to "tensorboard" \
    # --deepspeed configs/ds_config_zero3.json \
    # --bf16 True