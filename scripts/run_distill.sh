# sample scripts for running various knowledge distillation approaches
# we use resnet32x4 and resnet8x4 as an example

# CIFAR
# KD
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill kd --model_s resnet8x4 -c 1 -d 1 -b 0 --trial 0 --gpu_id 0
# FitNet
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill hint --model_s resnet8x4 -c 1 -d 1 -b 100 --trial 0 --gpu_id 0
# AT
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill attention --model_s resnet8x4 -c 1 -d 1 -b 1000 --trial 0 --gpu_id 0
# SP
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill similarity --model_s resnet8x4 -c 1 -d 1 -b 3000 --trial 0 --gpu_id 0
# VID
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill vid --model_s resnet8x4 -c 1 -d 1 -b 1 --trial 0 --gpu_id 0
# CRD
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill crd --model_s resnet8x4 -c 1 -d 1 -b 0.8 --trial 0 --gpu_id 0
# SemCKD
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill semckd --model_s resnet8x4 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0
# SRRL
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill srrl --model_s resnet8x4 -c 1 -d 1 -b 1 --trial 0 --gpu_id 0
# SimKD
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill simkd --model_s resnet8x4 -c 0 -d 0 -b 1 --trial 0 --gpu_id 0
# HKD
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --model_s resnet8x4 --distill hkd -c 1 -d 1 -m 1 -b 3 --trial 1
# GCKD
python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --model_s resnet8x4 --distill gckd --last_feature 1 --gnnlayer TAG --layers 2 --gnnencoder one --adj_k 8 --NPerturb 0.1 --EPerturb 0 -c 1 -d 1 -m 1 -b 3 --trial 1
# ImageNets
# SimKD
python train_student.py --path_t ./save/teachers/models/ResNet50_vanilla/ResNet50_best.pth --batch_size 256  --epochs 120 --dataset imagenet --model_s ResNet18 --distill simkd -c 0 -d 0 -b 1 --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 --num_workers 32 --gpu_id 0,1,2,3 --dist-url tcp://127.0.0.1:23344 --multiprocessing-distributed --dali gpu --trial 0
# GCKD
python train_student.py --path_t ./save/teachers/models/ResNet50_vanilla/ResNet50_best.pth --batch_size 256 --epochs 120 --dataset imagenet --model_s ResNet18 --distill gckd  --last_feature 1 --gnnlayer TAG --layers 2 --gnnencoder one --adj_k 8 --NPerturb 0.1 --EPerturb 0 -c 1 -d 1 -m 0 -b 3 -r 1 --trial 1 --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 --num_workers 8 --gpu_id 0 --multiprocessing-distributed --dali gpu --dist-url tcp://127.0.0.1:23444
