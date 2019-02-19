#!/bin/bash
GPU_ID=1
cur_date=`date +%Y-%m-%d-%H-%M-%S`
log_file_name=./examples/sketch_stroke/log/${cur_date}
./build/tools/caffe train \
    -solver ./examples/sketch_stroke/solver_next_stroke_final.prototxt \
    -gpu $GPU_ID 2>&1 | tee -a ${log_file_name}
