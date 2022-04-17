cd ../code
code_dir=../code
save_dir=../exp/Module_all

model_name=t5-large

### step1: train the module with synthetic data
# CUDA_VISIBLE_DEVICES=0 python train_module_all.py \
# --module_training_type para_all \
# --model_name_or_path $model_name \
# --input_join \
# --bs 20 --lr 3e-5 --epochs 1 --adafactor \
# --eval_epoch 0.05 --report_epoch 0.05 \
# --code_dir $code_dir \
# --exp_dir $save_dir/para_all \
# --save_model


### step2: train the module with Train-pseudo data
# CUDA_VISIBLE_DEVICES=0 python train_module_all.py \
# --module_training_type etree_all \
# --model_name_or_path $model_name \
# --resume_path ../exp/Module_all/<exp_name>/best_model.pth \
# --input_join \
# --bs 20 --lr 3e-5 --epochs 100 --adafactor \
# --eval_epoch 0.5 --report_epoch 0.5 \
# --code_dir $code_dir \
# --exp_dir $save_dir/para_etree_all \
# --save_model