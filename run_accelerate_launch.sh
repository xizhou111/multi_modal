log_dir='/mnt/cfs/NLP/zcl/multi_modal/logs_clip_L_chinese-roberta-wwm-ext_448_i2t_0811'
output_dir='/mnt/cfs/NLP/zcl/multi_modal/output_clip_L_chinese-roberta-wwm-ext_448_i2t_0811'

vit_model_name='clip_L'

resume_from_checkpoint='/mnt/cfs/NLP/zcl/multi_modal/output_clip_L_chinese-roberta-wwm-ext_448_i2t_0811/checkpoint-9000'

checkpoint_step=1000

accelerate launch --config_file ./default_config.yaml \
                                    train_accelerate_resume.py \
                                    --log_dir ${log_dir} \
                                    --output_dir ${output_dir} \
                                    --vit_model_name ${vit_model_name} \
                                    --resume_from_checkpoint ${resume_from_checkpoint} \
                                    --checkpoint_step ${checkpoint_step} \