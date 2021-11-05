#!/bin/bash
# 使用ex_num控制main函数里调用哪一个实验

ex_num=1
dataset_type=1
tcl=(0 1 2 3)
cuda=(5 3 3)
print_iter=(10 10 10)
epoch_num=(200 200 200)
for i in 0 1 2
do
  batch_size=256
  train_class=${tcl[${i}]}
  test_classes=(`expr ${tcl[${i}]} + 3`)
  CUDA_VISIBLE_DEVICES=${cuda[${i}]} nohup python -u main.py --batch_size=${batch_size} --print_iter=${print_iter[${i}]} --debug=False --ex_num=${ex_num} --dataset_type=${dataset_type} --train_class=${train_class} --test_classes=${test_classes} > log/ex${ex_num}_${dataset_type}_${train_class}.log 2>&1 &
done