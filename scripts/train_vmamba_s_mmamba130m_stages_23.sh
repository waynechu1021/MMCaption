OUTPUTDIR=vmamba-s-mmamba130m_stages_23_gluencoder_8query
RUNNAME=$(date '+%m-%d-%H-%M')-$(hostname)-${OUTPUTDIR}
wandb offline
CUDA_VISIBLE_DEVICES=6 python train.py \
    --seed 42 \
    --vmamba_path configs/vmamba_s_stage_23.json \
    --mamba_lm_path pretrained_model/language_model/mamba-130m \
    --max_length 60 \
    --output_dir output-seed/${OUTPUTDIR} \
    --tokenizer pretrained_model/gpt-neox-20b \
    --data_path data/coco2014 \
    --image_folder data/coco2014/images \
    --num_image_tokens 8 \
    --evaluate_file_path data/coco2014/annotations/coco_karpathy_val_gt.json \
    --num_train_epochs 5 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --dataloader_num_workers 2 \
    --report_to tensorboard wandb \
    --run_name ${RUNNAME} \