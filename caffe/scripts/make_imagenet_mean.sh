#!/usr/bin/env sh
# Compute the mean image from the input training lmdb

EXAMPLE=output/dataset/location
DATA=your/datalist/root/location
TOOLS=your/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/25_gqstroke_train_lmdb_1 \
	$DATA/25_gqstroke_train_mean.binaryproto

echo "Done."
