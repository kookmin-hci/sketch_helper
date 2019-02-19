import numpy as np 
import matplotlib.pyplot as plt
import lmdb
from PIL import Image
import caffe
import cv2
import pickle
from tqdm import tqdm

#Change paths to each files"
deployfile = 'yours/caffe/examples/sketch_stroke/deploy_next_stroke.prototxt'
weightfile = 'yours/next_stroke_snapshot7_1024_iter_4140000.caffemodel'
meanfile = 'yours/25_gqstroke_train_mean.binaryproto'
save_folder = 'yours/25_feature_map/'
list_folder = '/media/hci-gpu/Plextor1tb/google_quick_draw/stroke/train_set_data_list/'
img_folder = '/media/hci-gpu/Plextor1tb/google_quick_draw/stroke/imgData/'

#net
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(deployfile, weightfile, caffe.TEST)

print "net setting"
blob = caffe.proto.caffe_pb2.BlobProto()
data = open( meanfile, 'rb' ).read()
blob.ParseFromString(data)
mean = np.array( caffe.io.blobproto_to_array(blob) )

print "trans setting"
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data1': net.blobs['data1'].data.shape})
transformer.set_transpose('data1', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data1', mean[0])            # subtract the dataset-mean value in each channel
transformer.set_channel_swap('data1', (2,1,0))  # swap channels from RGB to BGR
transformer.set_input_scale('data1', 0.00390625)

print "let's start"
#hash map initialize
dataset = [] 
for j in tqdm(range(345)):
    for s in range(0,5):
        dataset = [] 
        failed_dataset = [] 
        f=open(list_folder+str(s)+"_"+str(j)+".txt",'r')
        count = 0
        for i in tqdm(range(20000)):
            line = f.readline()
            label = j
            
            filename = line[8:len(line)-len(str(j))-2]
            image = cv2.imread(img_folder + filename)

            transformed_image = transformer.preprocess('data1', image)

            net.blobs['data1'].data[...] = transformed_image

            if s==0: 
                net.blobs['raw_clip_data'].data[...] = 0
            else:
                net.blobs['raw_clip_data'].data[...] = 1

            out = net.forward()
            predicts = out['prob']

            predict = predicts.argmax()
            if int(label) == int(predict):
                feature_map = net.blobs['ip1'].data
                x = np.sign(feature_map)
                x_bit = np.maximum(x,0,x).astype(np.bool_) 
                x_binary = np.packbits(x_bit)
                temp = {'filename': filename, 'feature_map':x_bit}
                dataset.append(temp)
                count += 1
            else:
                feature_map = net.blobs['ip1'].data
                x = np.sign(feature_map)
                x_bit = np.maximum(x,0,x).astype(np.bool_) 
                x_binary = np.packbits(x_bit)
                temp = {'filename': filename, 'feature_map':x_bit}
                failed_dataset.append(temp)

            if count > 2000:
                break

        if count < 2000:
            print count, "need more data"
            dataset = dataset + failed_dataset[0:2000-count]

        f.close() 
        with open(save_folder+str(s)+"_"+str(label)+".list", 'ab') as sf:
                pickle.dump(dataset, sf, protocol=pickle.HIGHEST_PROTOCOL)

