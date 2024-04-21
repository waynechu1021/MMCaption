OUTPUTDIR=eval
RUNNAME=$(date '+%m-%d-%H-%M')-$(hostname)-${OUTPUTDIR}
wandb offline
CUDA_VISIBLE_DEVICES=4 python eval.py \
    --ckpt_path /ssd1/zhuweiye/Multimodal-Mamba/output-seed/deit-b-mmamba130m-gluencoder/checkpoint-56000 \
    --seed 42 \
    --output_dir output-seed/${OUTPUTDIR} \
    --num_image_tokens 196 \
    --evaluate_file_path data/coco2014/annotations/coco_karpathy_val_gt.json \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --dataloader_num_workers 2 \
    --report_to tensorboard \
    --run_name ${RUNNAME} \