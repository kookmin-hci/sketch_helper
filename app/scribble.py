#############################################################################
##
## Copyright (C) 2013 Riverbank Computing Limited.
## Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
## All rights reserved.
##
## This file is modified from examples of PyQt.
##
## $QT_BEGIN_LICENSE:BSD$
## You may use this file under the terms of the BSD license as follows:
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##   * Redistributions of source code must retain the above copyright
##     notice, this list of conditions and the following disclaimer.
##   * Redistributions in binary form must reproduce the above copyright
##     notice, this list of conditions and the following disclaimer in
##     the documentation and/or other materials provided with the
##     distribution.
##   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
##     the names of its contributors may be used to endorse or promote
##     products derived from this software without specific prior written
##     permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
## $QT_END_LICENSE$
##
#############################################################################

#UI import
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter


#Caffe import
import math
import numpy as np
import lmdb
from PIL import Image
import caffe
import cv2
import pickle
import timeit
import qimage2ndarray
import datetime, os
import copy

with_dataset = True
save_image = False
searchImage = ['a','a','a','a','a','a','a','a','a','a']
previous_can_img_list = []
prev_stroke = 0
img_folder = '/media/hci-gpu/Plextor1tb/google_quick_draw/stroke/imgData/'
new_folder = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') 
newpath = '/media/hci-gpu/Plextor1tb/Ourdata/'+ new_folder

if save_image:
    if not os.path.exists(newpath):
            os.makedirs(newpath)
img_count =0
search_count = 0


class MyCaffe(QThread):
    
    button_update_signal = pyqtSignal()
    image_update_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super(MyCaffe, self).__init__(parent)
        global with_dataset

        #net
        caffe.set_mode_gpu()
        caffe.set_device(0)
        self.net = caffe.Net('/home/hci-gpu/caffe/examples/sketch_stroke/deploy_next_stroke.prototxt','/storage/data/snapshot/next_stroke_snapshot7_1024_iter_4140000.caffemodel', caffe.TEST)

        #meanfile
        meanfile = '/media/hci-gpu/Plextor1tb/google_quick_draw/stroke/dataset/25_gqstroke_train_mean.binaryproto'
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open( meanfile, 'rb' ).read()
        blob.ParseFromString(data) 
        mean = np.array( caffe.io.blobproto_to_array(blob) )

        # create transformer for the input called 'data1'
        self.transformer = caffe.io.Transformer({'data1': self.net.blobs['data1'].data.shape})
        self.transformer.set_transpose('data1', (2,0,1))  # move image channels to outermost dimension
        self.transformer.set_mean('data1', mean[0])            # subtract the dataset-mean value in each channel
        self.transformer.set_channel_swap('data1', (2,1,0))  # swap channels from RGB to BGR
        self.transformer.set_input_scale('data1', 0.00390625)

        #hash map initialize
        self.dataset = [] 
        self.save_folder = '/media/hci-gpu/Plextor1tb/google_quick_draw/stroke/dataset/25_feature_map_s/'
	self.list_folder = '/media/hci-gpu/Plextor1tb/google_quick_draw/stroke/25000_data_list/'

	self.num_class = 345 
	self.num_stroke = 5

        #data load
        if with_dataset:
            self.num_check_class = [8,5,3,3,1]
            self.num_checked_image = [4,6,10,10,30]
            for s in range(self.num_stroke):
                for j in range(self.num_class): 
                    with open(self.save_folder+str(s)+"_"+str(j+1)+".list", 'rb') as sf:
                        listcursor = pickle.load(sf)
                    print "stroke"+str(s)+ " class"+str(j+1)
                    self.dataset.append(listcursor) 
        else:
            #fast running no dataset
            self.num_check_class = [10,10,10,10,10]
            self.num_checked_image = [10,10,10,10,10]

    def __del__(self):
        print("..........end thread.......")
        self.wait()

    @pyqtSlot(QImage, int)
    def queryImage(self, image, stroke, ):
        global with_dataset, previous_can_img_list, prev_stroke, searchImage, img_count, search_count, newpath
        #time check 
        start_time0 = timeit.default_timer()
	raw_image = qimage2ndarray.rgb_view(image)
        input_image  = cv2.resize(raw_image, (100,100), cv2.INTER_CUBIC) 
        if save_image :
            cv2.imwrite(newpath+'/'+str(img_count)+'_'+str(search_count)+".png",input_image) 

        transformed_image = self.transformer.preprocess('data1', input_image)
        self.net.blobs['data1'].data[...] = transformed_image

        if stroke == 0 :
            self.net.blobs['raw_clip_data'].data[...] = 0 # np.zeros(1)
        else:
            self.net.blobs['raw_clip_data'].data[...] = 1 # np.ones(1)

        out = self.net.forward()
        outnp = out['prob'][0]
        
	classlist = []
	for k in range(self.num_check_class[stroke]):
	    temp = outnp.argmax()
	    classlist.append(temp)
	    outnp[temp] = 0
		
	VQ = self.net.blobs['ip0'].data
	c_img_list = []
	value_list = []

        if with_dataset:
            can_img_list = []
            
            print "prev_stroke:", prev_stroke , ',' , 'stroke:', stroke
            if len(previous_can_img_list) >0:
                if stroke > prev_stroke:
                    for prev_can in previous_can_img_list:
                        prev_can[0] += 345* (stroke - prev_stroke)

                elif stroke < prev_stroke:
                    for prev_can in previous_can_img_list:
                        prev_can[0] -= 345* (prev_stroke - stroke)
            prev_stroke = stroke

            ##### sorting by MAX correlation
            ### New candidate list
            for candidate_class in classlist:
                p = stroke*self.num_class + candidate_class -1
                temp_list = []
                for i in range(1000):
                    
                    VIP = self.dataset[p][i]['feature_map']
                    temp = np.sum(np.logical_not(np.logical_xor(VQ,VIP)))
                    temp_list.append(temp)

                for i in range(self.num_checked_image[stroke]):
                    m = np.argmax(temp_list)
                    can_img_list.append([p, m, temp_list[m]])
                    temp_list[m] = 0
            # check previous list
            total_list = []
            if len(previous_can_img_list) > 0:
                for previous_can in previous_can_img_list:
                    equ_can = False
                    for del_idx, can_img in enumerate(can_img_list):
                        p = can_img[0]    
                        m = can_img[1]
                        val = can_img[2]
                        #when p, m exist increase val
                        if previous_can[0] == p  and previous_can[1] == m:
                            val += 0.05 * previous_can[2] * (stroke + 1)
                            total_list.append([p, m, val])
                            del can_img_list[del_idx]
                            equ_can = True
                            break
                        #when p, m non-exist add it 
                    if equ_can == False:
                        total_list.append(previous_can)
                can_img_list += total_list
           
            previous_can_img_list  = copy.deepcopy(can_img_list)
            ### Generate candidate file list
            for i in range(30):
                temp = max(can_img_list, key=lambda x: x[2])
                p = temp[0]
                index = temp[1]
                if stroke >= 4:
                    outfile_name = self.dataset[p][index]['filename']
                else:
                    # get next stroke (+345)
                    outfile_name = self.dataset[p+345][index]['filename']
                c_img_list.append(outfile_name)
                value_list.append(outfile_name)
                value_list.append(temp[2])
                max(can_img_list, key=lambda x: x[2])[2] = 0

        #### Fast running no Dataset
        if not with_dataset:
            c_img_list = [ '1/aircraft carrier_399_2.png','1/aircraft carrier_360_2.png','120/eyeglasses_1166_2.png' ,'121/face_62_3.png' ,'122/fan_426_3.png' ,'123/feather_626_2.png' ,'124/fence_862_4.png' ,'124/fence_877_2.png','130/flashlight_1093_2.png' ,'142/goatee_1093_2.png']

        output_image = self.image_mix(c_img_list)
        searchImage = c_img_list
        
        #time check 
        elapsed = timeit.default_timer() - start_time0
        print "len : ", len(c_img_list) #, "candidate : ", c_img_list
        print "candidate:", value_list
        print "stroke: ",stroke,", searching time:" ,elapsed

        self.button_update_signal.emit() 
        self.image_update_signal.emit(output_image) 

    def image_mix(self,img_list): 
        #background weighted sum algorithm
        idx = 0
        output_image = np.zeros((800,800),dtype=np.uint8)
        empty_image = output_image.copy()
        for n in list(reversed(img_list)):
            candidate = cv2.imread(img_folder+n,0)
            resize_candidate  = cv2.resize(candidate, (800,800), cv2.INTER_CUBIC) 
            blur_candidate = cv2.GaussianBlur(resize_candidate,(21,21),0) 
            inv_candidate = cv2.bitwise_not(blur_candidate)
            if len(img_list) >10:
                #16~30 images
                if idx < len(img_list)/2:
                    temp_image = cv2.addWeighted(empty_image, 0.85, inv_candidate, 0.05,0)
                    output_image = cv2.add(output_image, temp_image)
                #6~15
                elif idx >= len(img_list)/2 and idx < len(img_list)*0.84:
                    temp_image = cv2.addWeighted(empty_image, 0.85, inv_candidate, 0.10,0)
                    output_image = cv2.add(output_image, temp_image)
                #1~5
                else:
                    temp_image = cv2.addWeighted(empty_image, 0.85, inv_candidate, 0.15,0)
                    output_image = cv2.add(output_image, temp_image)

            else:
                temp_image = cv2.addWeighted(empty_image, 0.85, inv_candidate, 0.15,0)
                output_image = cv2.add(output_image, temp_image)

            idx = idx+1

	output_image = cv2.bitwise_not(output_image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)

        return output_image


class ScribbleArea(QWidget):
    queryImage_signal = pyqtSignal(QImage, int)

    def __init__(self, parent=None):
        super(ScribbleArea, self).__init__(parent)
        self.myCaffe = MyCaffe(parent=self) 

        self.setAttribute(Qt.WA_StaticContents)
        self.modified = False
        self.triggered = False
        self.scribbling = False
        self.myPenWidth = 7
        self.myPenColor = Qt.black
        self.image = QImage(800,800, QImage.Format_ARGB32_Premultiplied)
        self.image.fill(qRgba(255, 255, 255, 0))
        
        self.bgimage = QImage(800,800, QImage.Format_ARGB32_Premultiplied)
        self.bgimage.fill(qRgba(255, 255, 255, 255))

        self.lastPoint = QPoint()
        self.isPen = True
        self.stLength = 0
        self.erLength = 0
        self.strokeStack = 0
        self.searchTrigger = 250
        self.strokeTrigger = 500
        self.stroke = 0
        self.btnCheckList = [False, False, False, False, False, False, False, False, False, False] 
        self.queryImage_signal.connect(self.myCaffe.queryImage)

    def openImage(self, fileName):
        loadedImage = QImage()
        if not loadedImage.load(fileName):
            return False

        newSize = loadedImage.size().expandedTo(self.size())
        self.resizeImage(loadedImage, newSize)
        self.image = loadedImage
        self.modified = False
        self.update()
        return True

    def saveImage(self, fileName, fileFormat):
        visibleImage = self.image
        self.resizeImage(visibleImage, self.size())

        if visibleImage.save(fileName, fileFormat):
            self.modified = False
            return True
        else:
            return False

    def setPenColor(self, newColor):
        self.myPenColor = newColor

    def setPenWidth(self, newWidth):
        self.myPenWidth = newWidth

    def clearImage(self):
        global searchImage, previous_can_img_list, prev_stroke
        previous_can_img_list = []
        prev_stroke = 0
        searchImage = ['a','a','a','a','a','a','a','a','a','a']
        self.btnCheckList = [False, False, False, False, False, False, False, False, False, False] 
        self.stroke = 0
        self.stLength = 0
        self.erLength = 0
        self.strokeStack = 0
        self.bgimage.fill(qRgba(255, 255, 255, 255))
        self.image.fill(qRgba(255, 255, 255, 0))
        self.modified = True
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.scribbling = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.scribbling:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.scribbling:
            self.drawLineTo(event.pos())
            self.scribbling = False

    def paintEvent(self, event):
        painter = QPainter(self)
        dirtyRect = event.rect()
        painter.drawImage(dirtyRect, self.bgimage, dirtyRect)
        painter.drawImage(dirtyRect, self.image, dirtyRect)

    def resizeEvent(self, event):
        if self.width() > self.image.width() or self.height() > self.image.height():
            self.resizeImage(self.image, QSize(800, 800))
            self.update()

        if self.width() > self.bgimage.width() or self.height() > self.bgimage.height():
            self.resizeImage(self.bgimage, QSize(800, 800))
            self.update()

        super(ScribbleArea, self).resizeEvent(event)

    @pyqtSlot(np.ndarray)
    def updateImage(self, output_image):
        self.bgimage = qimage2ndarray.array2qimage(output_image)
        self.update()

    def drawLineTo(self, endPoint):
        newLineLength = 0
        #pen setting
        if self.isPen:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.myPenColor, self.myPenWidth, Qt.SolidLine,
                    Qt.RoundCap, Qt.RoundJoin))

            painter.drawLine(self.lastPoint, endPoint)
            newLineLength = math.sqrt((self.lastPoint.x()-endPoint.x())*(self.lastPoint.x()-endPoint.x()) 
                    +(self.lastPoint.y()-endPoint.y())*(self.lastPoint.y()-endPoint.y()))
            
            self.stLength = self.stLength + newLineLength
            if self.erLength > self.strokeTrigger and all(item == False for item in self.btnCheckList):
                self.stroke = self.stroke - int(self.erLength/self.strokeTrigger)
                if self.stroke < 0:
                    self.stroke = 0
                self.erLength = 0

            if (self.stLength > self.searchTrigger)  and all(item == False for item in self.btnCheckList):
                self.queryImage_signal.emit(self.image, self.stroke)
                self.strokeStack = self.strokeStack+ self.stLength
                self.stLength = 0
                if self.strokeStack > self.strokeTrigger:
                    if self.stroke < 4:
                        self.stroke = self.stroke + 1
                    self.strokeStack = 0
            self.modified = True
            self.update()

        #erasing function
        else:
            #Qimage cursor position to alpha zero update
            newLineLength = math.sqrt((self.lastPoint.x()-endPoint.x())*(self.lastPoint.x()-endPoint.x()) 
                    +(self.lastPoint.y()-endPoint.y())*(self.lastPoint.y()-endPoint.y()))
            self.erLength = self.erLength + newLineLength
            self.stLength = 0
            #sorting endpoint and lastpoint
            if self.lastPoint.x() < endPoint.x():
                x1 = self.lastPoint.x()
                y1 = self.lastPoint.y()
                x2 = endPoint.x()
                y2 = endPoint.y()

            else:
                x2 = self.lastPoint.x()
                y2 = self.lastPoint.y()
                x1 = endPoint.x()
                y1 = endPoint.y()
            #set boundary
            if x1 - self.myPenWidth/2 >= 0 and x1 +self.myPenWidth/2 < 800 \
               and x2 - self.myPenWidth/2 >= 0 and x2 +self.myPenWidth/2 < 800 \
               and y1 - self.myPenWidth/2 >= 0 and y1 +self.myPenWidth/2 < 800 \
               and y2 - self.myPenWidth/2 >= 0 and y2 +self.myPenWidth/2 < 800 :

                if x1 != x2 :
                    m = float(y2-y1)/float(x2-x1)
                    for i in range(x1,x2):
                        for j in range(-1*self.myPenWidth/2, self.myPenWidth/2):
                            for k in range(-1*self.myPenWidth/2, self.myPenWidth/2):
                                self.image.setPixel(i+j,int(m*(i-x1))+y1+k, QColor(255,255,255,0).rgba())
                elif x1 == x2 and y1 != y2:
                    for i in range(min(y1,y2),max(y1,y2)):
                        for j in range(-1*self.myPenWidth/2, self.myPenWidth/2):
                            for k in range(-1*self.myPenWidth/2, self.myPenWidth/2):
                                self.image.setPixel(x1+j,i+k, QColor(255,255,255,0).rgba())
                else:
                    for j in range(-1*self.myPenWidth/2, self.myPenWidth/2):
                        for k in range(-1*self.myPenWidth/2, self.myPenWidth/2):
                            self.image.setPixel(x1+j,y1+k, QColor(255,255,255,0).rgba())

                rad = self.myPenWidth / 2 + 2
                self.update(QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))

        self.lastPoint = QPoint(endPoint)

    def resizeImage(self, image, newSize):
        if image.size() == newSize:
            return

        newImage = QImage(newSize, QImage.Format_ARGB32_Premultiplied)
        newImage.fill(qRgb(255, 255, 255))
        painter = QPainter(newImage)
        painter.drawImage(QPoint(0, 0), image)
        self.image = newImage

    def print_(self):
        printer = QPrinter(QPrinter.HighResolution)

        printDialog = QPrintDialog(printer, self)
        if printDialog.exec_() == QPrintDialog.Accepted:
            painter = QPainter(printer)
            rect = painter.viewport()
            size = self.image.size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.image.rect())
            painter.drawImage(0, 0, self.image)
            painter.end()

    def isModified(self):
        return self.modified

    def penColor(self):
        return self.myPenColor

    def penWidth(self):
        return self.myPenWidth

    def changePen(self):
        if self.isPen:
            #Pen width should be even. because of erase method
            self.myPenWidth = 30 
            self.myPenColor = Qt.white
            self.isPen = False
        else:
            self.myPenWidth = 7
            self.myPenColor = Qt.black
            self.isPen = True

    def icon2canvas(self, icon_image, check):
        if self.btnCheckList[check]:
            self.btnCheckList[check] = False
        else:
            self.btnCheckList[check] = True

        self.bgimage.fill(qRgba(255, 255, 255, 255))

        enableBtnList = []
        for idx, i in enumerate(self.btnCheckList):
            if i and icon_image != 'a':
                enableBtnList.append(searchImage[idx])

        if len(enableBtnList) > 0:
            print enableBtnList
            temp_img = self.myCaffe.image_mix(enableBtnList)
            resize_img  = cv2.resize(temp_img, (800,800), cv2.INTER_CUBIC) 
            blur_img = cv2.GaussianBlur(resize_img,(21,21),0) 
            self.bgimage = qimage2ndarray.array2qimage(blur_img)
        else:
            self.queryImage_signal.emit(self.image, self.stroke)
            
        self.update()


class FunctionArea(QWidget):
    
    def __init__(self, parent = None):
        super(FunctionArea, self).__init__(parent)
        self.initUI()
        
    def initUI(self):
       
        self.funcArea = QGridLayout()
        self.funcArea.setSpacing(10)

        self.change = QPushButton("Pen : Pen/Eraser")
        self.clear = QPushButton("Clear board")

        self.change.setFixedWidth(170)
        self.change.setFixedHeight(30)
        self.clear.setFixedWidth(170)
        self.clear.setFixedHeight(30)

        self.funcArea.addWidget(self.change, 0,0)
        self.funcArea.addWidget(self.clear, 0,1)

        self.setLayout(self.funcArea) 


class OutArea(QWidget):
    
    def __init__(self, parent = None):
        super(OutArea, self).__init__(parent)
        self.initUI()

    def firstIcon(self):
        self.b1.setIcon(QIcon(QPixmap('data/init/1.png')))
        self.b2.setIcon(QIcon(QPixmap('data/init/2.png')))
        self.b3.setIcon(QIcon(QPixmap('data/init/3.png')))
        self.b4.setIcon(QIcon(QPixmap('data/init/4.png')))
        self.b5.setIcon(QIcon(QPixmap('data/init/5.png')))
        self.b6.setIcon(QIcon(QPixmap('data/init/6.png')))
        self.b7.setIcon(QIcon(QPixmap('data/init/7.png')))
        self.b8.setIcon(QIcon(QPixmap('data/init/8.png')))
        self.b9.setIcon(QIcon(QPixmap('data/init/9.png')))
        self.b10.setIcon(QIcon(QPixmap('data/init/10.png')))

        self.b1.setChecked(False)
        self.b2.setChecked(False)
        self.b3.setChecked(False)
        self.b4.setChecked(False)
        self.b5.setChecked(False)
        self.b6.setChecked(False)
        self.b7.setChecked(False)
        self.b8.setChecked(False)
        self.b9.setChecked(False)
        self.b10.setChecked(False)
    
    def updateIcon(self):
        global searchImage
        self.b1.setIcon(QIcon(QPixmap(img_folder+searchImage[0]).scaled(150,150,Qt.KeepAspectRatio)))
        self.b2.setIcon(QIcon(QPixmap(img_folder+searchImage[1]).scaled(150,150,Qt.KeepAspectRatio)))
        self.b3.setIcon(QIcon(QPixmap(img_folder+searchImage[2]).scaled(150,150,Qt.KeepAspectRatio)))
        self.b4.setIcon(QIcon(QPixmap(img_folder+searchImage[3]).scaled(150,150,Qt.KeepAspectRatio)))
        self.b5.setIcon(QIcon(QPixmap(img_folder+searchImage[4]).scaled(150,150,Qt.KeepAspectRatio)))
        self.b6.setIcon(QIcon(QPixmap(img_folder+searchImage[5]).scaled(150,150,Qt.KeepAspectRatio)))
        self.b7.setIcon(QIcon(QPixmap(img_folder+searchImage[6]).scaled(150,150,Qt.KeepAspectRatio)))
        self.b8.setIcon(QIcon(QPixmap(img_folder+searchImage[7]).scaled(150,150,Qt.KeepAspectRatio)))
        self.b9.setIcon(QIcon(QPixmap(img_folder+searchImage[8]).scaled(150,150,Qt.KeepAspectRatio)))
        self.b10.setIcon(QIcon(QPixmap(img_folder+searchImage[9]).scaled(150,150,Qt.KeepAspectRatio)))

    def initUI(self):
        grid = QGridLayout()

        self.b1 = QPushButton()
        self.b1.setIconSize(QSize(150,150))
        self.b1.setCheckable(True)
        self.b1.toggle()

        self.b2 = QPushButton()
        self.b2.setIconSize(QSize(150,150))
        self.b2.setCheckable(True)
        self.b2.toggle()
        
        self.b3 = QPushButton()
        self.b3.setIconSize(QSize(150,150))
        self.b3.setCheckable(True)
        self.b3.toggle()

        self.b4 = QPushButton()
        self.b4.setIconSize(QSize(150,150))
        self.b4.setCheckable(True)
        self.b4.toggle()
        
        self.b5 = QPushButton()
        self.b5.setIconSize(QSize(150,150))
        self.b5.setCheckable(True)
        self.b5.toggle()

        self.b6 = QPushButton()
        self.b6.setIconSize(QSize(150,150))
        self.b6.setCheckable(True)
        self.b6.toggle()

        self.b7 = QPushButton()
        self.b7.setIconSize(QSize(150,150))
        self.b7.setCheckable(True)
        self.b7.toggle()

        self.b8 = QPushButton()
        self.b8.setIconSize(QSize(150,150))
        self.b8.setCheckable(True)
        self.b8.toggle()

        self.b9 = QPushButton()
        self.b9.setIconSize(QSize(150,150))
        self.b9.setCheckable(True)
        self.b9.toggle()

        self.b10 = QPushButton()
        self.b10.setIconSize(QSize(150,150))
        self.b10.setCheckable(True)
        self.b10.toggle()

        self.b1.setIcon(QIcon(QPixmap('data/init/1.png')))
        self.b2.setIcon(QIcon(QPixmap('data/init/2.png')))
        self.b3.setIcon(QIcon(QPixmap('data/init/3.png')))
        self.b4.setIcon(QIcon(QPixmap('data/init/4.png')))
        self.b5.setIcon(QIcon(QPixmap('data/init/5.png')))
        self.b6.setIcon(QIcon(QPixmap('data/init/6.png')))
        self.b7.setIcon(QIcon(QPixmap('data/init/7.png')))
        self.b8.setIcon(QIcon(QPixmap('data/init/8.png')))
        self.b9.setIcon(QIcon(QPixmap('data/init/9.png')))
        self.b10.setIcon(QIcon(QPixmap('data/init/10.png')))

        self.b1.setFixedWidth(160)
        self.b1.setFixedHeight(160)
        self.b2.setFixedWidth(160)
        self.b2.setFixedHeight(160)
        self.b3.setFixedWidth(160)
        self.b3.setFixedHeight(160)
        self.b4.setFixedWidth(160)
        self.b4.setFixedHeight(160)
        self.b5.setFixedWidth(160)
        self.b5.setFixedHeight(160)
        self.b6.setFixedWidth(160)
        self.b6.setFixedHeight(160)
        self.b7.setFixedWidth(160)
        self.b7.setFixedHeight(160)
        self.b8.setFixedWidth(160)
        self.b8.setFixedHeight(160)
        self.b9.setFixedWidth(160)
        self.b9.setFixedHeight(160)
        self.b10.setFixedWidth(160)
        self.b10.setFixedHeight(160)

        grid.addWidget(self.b1, 0,0)
        grid.addWidget(self.b2, 0,1)
        grid.addWidget(self.b3, 1,0)
        grid.addWidget(self.b4, 1,1)
        grid.addWidget(self.b5, 2,0)
        grid.addWidget(self.b6, 2,1)
        grid.addWidget(self.b7, 3,0)
        grid.addWidget(self.b8, 3,1)
        grid.addWidget(self.b9, 4,0)
        grid.addWidget(self.b10, 4,1)

        self.setLayout(grid) 


class MainLayout(QWidget):
    
    def __init__(self, parent = None):
        super(MainLayout, self).__init__(parent)
        self.initUI()
        
    def initUI(self):
        canvas = QLabel('Canvas', self)
        canvas.setFixedWidth(800)
        canvas.setFixedHeight(10)
        result = QLabel('Search result', self)
        result.setFixedWidth(370)
        result.setFixedHeight(10)

        font = QFont()
        font.setPointSize(10)
        canvas.setFont(font)
        result.setFont(font)
	self.scribbleArea = ScribbleArea(self)
        self.outArea = OutArea(self)	
        self.funcArea = FunctionArea(self)

        canvas.setGeometry(10,10,790,100)
        result.setGeometry(820,10,380,100)
        self.scribbleArea.setGeometry(10,47,800,800)
        self.outArea.setGeometry(810,30,390,900)
        self.funcArea.setGeometry(10,850,800,80)
   
        #add action
        self.funcArea.clear.clicked.connect(self.scribbleArea.clearImage)
        self.funcArea.clear.clicked.connect(self.outArea.firstIcon)

        self.funcArea.change.clicked.connect(self.scribbleArea.changePen)
        self.funcArea.change.clicked.connect(self.change_text)

        global searchImage

        self.outArea.b1.clicked.connect(lambda:self.scribbleArea.icon2canvas(searchImage[0],0))
        self.outArea.b2.clicked.connect(lambda:self.scribbleArea.icon2canvas(searchImage[1],1))
        self.outArea.b3.clicked.connect(lambda:self.scribbleArea.icon2canvas(searchImage[2],2))
        self.outArea.b4.clicked.connect(lambda:self.scribbleArea.icon2canvas(searchImage[3],3))
        self.outArea.b5.clicked.connect(lambda:self.scribbleArea.icon2canvas(searchImage[4],4))
        self.outArea.b6.clicked.connect(lambda:self.scribbleArea.icon2canvas(searchImage[5],5))
        self.outArea.b7.clicked.connect(lambda:self.scribbleArea.icon2canvas(searchImage[6],6))
        self.outArea.b8.clicked.connect(lambda:self.scribbleArea.icon2canvas(searchImage[7],7))
        self.outArea.b9.clicked.connect(lambda:self.scribbleArea.icon2canvas(searchImage[8],8))
        self.outArea.b10.clicked.connect(lambda:self.scribbleArea.icon2canvas(searchImage[9],9))
      
        self.scribbleArea.myCaffe.button_update_signal.connect(self.outArea.updateIcon)
        self.scribbleArea.myCaffe.image_update_signal.connect(self.scribbleArea.updateImage)

        self.setGeometry(0, 0, 1200, 950)

    def change_text(self):
        if self.scribbleArea.isPen:
            self.funcArea.change.setText("Pen : Pen/Eraser")
        else:
            self.funcArea.change.setText("Eraser : Pen/Eraser")


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.saveAsActs = []
        self.initUI()
        self.setCursor(QCursor(Qt.CrossCursor))

    def initUI(self):
	
        self.layout = MainLayout()
	self.setCentralWidget(self.layout)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Sketch helper")
        self.setFixedSize(1200,950)

    def closeEvent(self, event):
        if self.maybeSave():
            event.accept()
        else:
            event.ignore()

    def open(self):
        if self.maybeSave():
            fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                    QDir.currentPath())
            if fileName:
                self.layout.scribbleArea.openImage(fileName)

    def save(self):
        action = self.sender()
        fileFormat = action.data()
        self.saveFile(fileFormat)

    def penColor(self):
        newColor = QColorDialog.getColor(self.layout.scribbleArea.penColor())
        if newColor.isValid():
            self.layout.scribbleArea.setPenColor(newColor)

    def penWidth(self):
        newWidth, ok = QInputDialog.getInt(self, "Scribble",
                "Select pen width:", self.layout.scribbleArea.penWidth(), 1, 50, 1)
        if ok:
            self.layout.scribbleArea.setPenWidth(newWidth)

    def about(self):
        QMessageBox.about(self, "About Scribble",
                "<p>The <b>Scribble</b> example shows how to use "
                "QMainWindow as the base widget for an application, and how "
                "to reimplement some of QWidget's event handlers to receive "
                "the events generated for the application's widgets:</p>"
                "<p> We reimplement the mouse event handlers to facilitate "
                "drawing, the paint event handler to update the application "
                "and the resize event handler to optimize the application's "
                "appearance. In addition we reimplement the close event "
                "handler to intercept the close events before terminating "
                "the application.</p>"
                "<p> The example also demonstrates how to use QPainter to "
                "draw an image in real time, as well as to repaint "
                "widgets.</p>")

    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O",
                triggered=self.open)

        for format in QImageWriter.supportedImageFormats():
            format = str(format)

            text = format.upper() + "..."

            action = QAction(text, self, triggered=self.save)
            action.setData(format)
            self.saveAsActs.append(action)

        self.printAct = QAction("&Print...", self,
                triggered=self.layout.scribbleArea.print_)

        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                triggered=self.close)

        self.penColorAct = QAction("&Pen Color...", self,
                triggered=self.penColor)

        self.penWidthAct = QAction("Pen &Width...", self,
                triggered=self.penWidth)

        self.clearScreenAct = QAction("&Clear Screen", self, shortcut="Ctrl+L",
                triggered=self.layout.scribbleArea.clearImage)

        self.aboutAct = QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QAction("About &Qt", self,
                triggered=QApplication.instance().aboutQt)

    def createMenus(self):
        self.saveAsMenu = QMenu("&Save As", self)
        for action in self.saveAsActs:
            self.saveAsMenu.addAction(action)

        fileMenu = QMenu("&File", self)
        fileMenu.addAction(self.openAct)
        fileMenu.addMenu(self.saveAsMenu)
        fileMenu.addAction(self.printAct)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAct)

        optionMenu = QMenu("&Options", self)
        optionMenu.addAction(self.penColorAct)
        optionMenu.addAction(self.penWidthAct)
        optionMenu.addSeparator()
        optionMenu.addAction(self.clearScreenAct)

        helpMenu = QMenu("&Help", self)
        helpMenu.addAction(self.aboutAct)
        helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(fileMenu)
        self.menuBar().addMenu(optionMenu)
        self.menuBar().addMenu(helpMenu)

    def maybeSave(self):
        if self.layout.scribbleArea.isModified():
            ret = QMessageBox.warning(self, "Scribble",
                        "The image has been modified.\n"
                        "Do you want to save your changes?",
                        QMessageBox.Save | QMessageBox.Discard |
                        QMessageBox.Cancel)
            if ret == QMessageBox.Save:
                return self.saveFile('png')
            elif ret == QMessageBox.Cancel:
                return False

        return True

    def saveFile(self, fileFormat):
        initialPath = QDir.currentPath() + '/untitled.' + fileFormat

        fileName, _ = QFileDialog.getSaveFileName(self, "Save As", initialPath,
                "%s Files (*.%s);;All Files (*)" % (fileFormat.upper(), fileFormat))
        if fileName:
            return self.layout.scribbleArea.saveImage(fileName, fileFormat)

        return False


if __name__ == '__main__':

    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
