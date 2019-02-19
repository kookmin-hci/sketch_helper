#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=output/dataset/location
DATA=your/datalist/root/location
TOOLS=your/caffe/build/tools

TRAIN_DATA_ROOT=your/image/data/location
VAL_DATA_ROOT=your/image/data/location

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

echo "Creating train lmdb 01234..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
	--encoded \
    $TRAIN_DATA_ROOT \
    $DATA/75000_Train_stroke.txt \
    $EXAMPLE/25_gqstroke_train_lmdb_1

echo "Creating second train lmdb 12344..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
	--encoded \
    $TRAIN_DATA_ROOT \
    $DATA/75000_Train_stroke2.txt \
    $EXAMPLE/25_gqstroke_train_lmdb_2

echo "Creating Test lmdb.."

LOG_logtostderr=1 $TOOLS/convert_imageset \
	--encoded \
    $VAL_DATA_ROOT \
    $DATA/75000_Test_stroke.txt \
    $EXAMPLE/25_gqstroke_test_lmdb

echo "Done."

echo "Creating Validation lmdb.."

LOG_logtostderr=1 $TOOLS/convert_imageset \
	--encoded \
    $VAL_DATA_ROOT \
	$DATA/75000_Val_stroke.txt \
    $EXAMPLE/25_gqstroke_val_lmdb

echo "Done."
