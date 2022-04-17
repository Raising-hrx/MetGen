cd ../code
data_dir=../data/Controller_data


##### Task 1
### Step1: make controller training data based on the orginal dataset and the trained module
# modify the arguments in make_controller_data.py and run it
# CUDA_VISIBLE_DEVICES=0  python make_controller_data.py 


### Step2: train the controller
# CUDA_VISIBLE_DEVICES=0 python train_Controller.py \
# --task_name task1 \
# --train_data_file $data_dir/train.controller.task1.v36.jsonl \
# --dev_data_file $data_dir/dev.controller.task1.v36.jsonl \
# --model_name_or_path albert-xxlarge-v2 \
# --bs 5 --lr 1e-5 --epochs 1000 --adafactor \
# --eval_epoch 50 --report_epoch 1 \
# --code_dir ../code \
# --exp_dir ../exp/Controller_task1 \
# --save_model --seed 2171





##### Task 2 & Task 3
### Step1: make controller training data based on the orginal dataset and the trained module
# modify the arguments in make_controller_data.py and run it
# CUDA_VISIBLE_DEVICES=0  python make_controller_data.py 

### Step2: train the controller
CUDA_VISIBLE_DEVICES=0 python train_Controller.py \
--task_name task2 \
--train_data_file $data_dir/train.controller.task2.v36.jsonl \
--dev_data_file $data_dir/dev.controller.task2.v36.jsonl \
--model_name_or_path albert-xxlarge-v2 \
--bs 4 --lr 1e-5 --epochs 1000 --adafactor \
--eval_epoch 25 --report_epoch 1 \
--code_dir ../code \
--exp_dir ../exp/Controller_task2 \
--save_model --seed 1260