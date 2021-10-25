#!/bin/bash
# 使用ex_num控制main函数里调用哪一个实验

ex_num=1
dataset_type=3
tcl=(0)
cuda=(5)
for i in 0
do
  batch_size=256
  train_class=${tcl[${i}]}
  CUDA_VISIBLE_DEVICES=${cuda[${i}]} nohup python -u main.py --batch_size=${batch_size} --debug=False --ex_num=${ex_num} --dataset_type=${dataset_type} --train_class=${train_class} > log/ex${ex_num}_${dataset_type}_${train_class}.log 2>&1 &
done


ex_num=2
dataset_type=3
tcl=(0)
cuda=(5)
for i in 0
do
  batch_size=256
  train_class=${tcl[${i}]}
  CUDA_VISIBLE_DEVICES=${cuda[${i}]} nohup python -u main.py --batch_size=${batch_size} --debug=False --ex_num=${ex_num} --dataset_type=${dataset_type} --train_class=${train_class} > log/ex${ex_num}_${dataset_type}_${train_class}.log 2>&1 &
done


ex_num=3
dataset_type=3
tcl=(0)
cuda=(5)
for i in 0
do
  batch_size=256
  train_class=${tcl[${i}]}
  CUDA_VISIBLE_DEVICES=${cuda[${i}]} nohup python -u main.py --batch_size=${batch_size} --debug=False --ex_num=${ex_num} --dataset_type=${dataset_type} --train_class=${train_class} > log/ex${ex_num}_${dataset_type}_${train_class}.log 2>&1 &
done


ex_num=4
dataset_type=3
tcl=(0)
cuda=(4)
for i in 0
do
  batch_size=256
  train_class=${tcl[${i}]}
  CUDA_VISIBLE_DEVICES=${cuda[${i}]} nohup python -u main.py --batch_size=${batch_size} --debug=False --ex_num=${ex_num} --dataset_type=${dataset_type} --train_class=${train_class} > log/ex${ex_num}_${dataset_type}_${train_class}.log 2>&1 &
done


ex_num=5
dataset_type=3
tcl=(0)
cuda=(4)
for i in 0
do
  batch_size=256
  train_class=${tcl[${i}]}
  CUDA_VISIBLE_DEVICES=${cuda[${i}]} nohup python -u main.py --batch_size=${batch_size} --debug=False --ex_num=${ex_num} --dataset_type=${dataset_type} --train_class=${train_class} > log/ex${ex_num}_${dataset_type}_${train_class}.log 2>&1 &
done
