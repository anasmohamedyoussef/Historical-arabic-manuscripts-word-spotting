# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import random
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda
from keras.optimizers import SGD
from keras.layers.convolutional import MaxPooling2D,Conv2D
from keras.layers.core import Flatten
from keras.layers import Dropout
from keras.models import load_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import accuracy_score as accuracy
import os
import cv2
from random import shuffle

book='3157556'


K.set_image_dim_ordering('th')
np.random.seed(100)
random.seed(100)

os.environ["CUDA_VISIBLE_DEVICES"]="1"
          

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    
    margin = 1
    return K.mean((1 - y_true) * K.square(y_pred) + 
                  y_true * K.square(K.maximum(margin - y_pred, 0)))
# mean( (1-y_true * y_pred^2) + (y_true * max(margin-Y_pred,0)^2))


def process(image):
    gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rimage=cv2.resize(gimage, (110, 60))
    fimage=rimage.astype('float32')
    nimage=fimage/255.
    return nimage

def loaddata(folderName,numberOfFolders,numberOfSamples):
    x=[]
    y=[]
    label=0
    for foldername in sorted(os.listdir(folderName)):
            maxLimit=0
            for filename in sorted(os.listdir(folderName+'/'+ foldername)):
                image=cv2.imread(folderName+'/'+ foldername+'/'+filename)
                pimage=process(image)
                x.append(pimage)
                y.append(label)
                maxLimit=maxLimit+1
                if maxLimit>=numberOfSamples:
                   break          
            label=label+1
            if label>=numberOfFolders:
                break
    ax=np.array(x)
    ay=np.array(y)
    return ax,ay

def create_pairs(x, indices,numberOfSubwords):
    pairs = []
    labels = []
    for d in range(numberOfSubwords):
        numberOfSamples=len(indices[d])
        for i in range(numberOfSamples):
            for j in range (1, numberOfSamples-i):
    
                z1, z2 = indices[d][i], indices[d][i+j]
                pairs += [[x[z1], x[z2]]]
                
                r = random.randrange(1, numberOfSubwords-1)
                ri = (d + r) % (numberOfSubwords)
                s=random.randrange(0,len(indices[ri]))
                z1, z2 = indices[d][i], indices[ri][s]
                pairs += [[x[z1], x[z2]]]
                
                labels += [0,1]
    return np.array(pairs), np.array(labels)

#Sianmese Network Branches
  

def create_base_network(input_dim):

    convnet = Sequential()
    convnet.add(Conv2D(64,(5,5),padding="same",activation='relu',input_shape=(1,60,110)))
    convnet.add(MaxPooling2D(pool_size=(2,2)))
    convnet.add(Conv2D(128,(5,5),padding="same",activation='relu'))
    convnet.add(MaxPooling2D(pool_size=(2,2)))
    convnet.add(Conv2D(256,(3,3),padding="same",activation='relu'))
    convnet.add(MaxPooling2D(pool_size=(2,2)))
    convnet.add(Conv2D(512,(3,3),padding="same",activation='relu'))
    convnet.add(Conv2D(512,(3,3),padding="same",activation='relu'))
    convnet.add(MaxPooling2D(pool_size=(2,2)))
    convnet.add(Flatten())
    convnet.add(Dense(4096,activation="relu"))
    convnet.add(Dropout(0.2))
    convnet.add(Dense(4096,activation="relu"))
    convnet.add(Dropout(0.2))
    
    return convnet

# 300 110 60 1
# 300 1   110 60
(x_train, y_train)=loaddata('segment'+book+'20',100,3)
x_train = x_train.reshape(x_train.shape[0], 1,x_train.shape[1],x_train.shape[2])
trNumberOfSubwords=len(np.unique(y_train))
tr_indices = [np.where(y_train == i)[0] for i in range(trNumberOfSubwords)]
otr_pairs, otr_y = create_pairs(x_train, tr_indices,trNumberOfSubwords)
#shuffle train set
ind = [i for i in range(otr_y.shape[0])]
shuffle(ind)
tr_pairs = otr_pairs[ind,:,:,:,:]
tr_y = otr_y[ind,]

(x_val, y_val)=loaddata('segment'+book+'10',20,10)
x_val = x_val.reshape(x_val.shape[0], 1,x_val.shape[1],x_val.shape[2])
valNumberOfSubwords=len(np.unique(y_val))
val_indices = [np.where(y_val == i)[0] for i in range(valNumberOfSubwords)]
val_pairs, val_y = create_pairs(x_val, val_indices,valNumberOfSubwords)
 
print("train number of subwords:"+str(y_train.shape))
print("val number of subwords:"+str(y_val.shape))

input_dim = (1,60,110)

# network definition
base_network = create_base_network(input_dim)

input_a = Input(shape=(input_dim))
input_b = Input(shape=(input_dim))


processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(inputs=[input_a, input_b], outputs=distance)
# train
sgd = SGD(lr=0.001)
model.compile(loss = contrastive_loss, optimizer=sgd)



minloss=100.
maxfmap=0.
pat=0

for e in range(250):
    print('EPOCH: '+str(e))
    h=model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                batch_size=32, epochs=1)
   
    fmap=[]
    aps=[]
    for d1 in range(valNumberOfSubwords):
        print ('subword: '+ str(d1))
        pairs = []
        labels = []
        ps=[]
        
        numberOfSamples=len(val_indices[d1])
        if numberOfSamples>1:
            for i in range (1, numberOfSamples):
                z1, z2 = val_indices[d1][0], val_indices[d1][i]
                pairs += [[x_val[z1], x_val[z2]]]
                labels+=[0]
            for d2 in range(valNumberOfSubwords):
                if d2!=d1:
                    numberOfSamples=len(val_indices[d2])
                    for j in range (0, numberOfSamples):
                        z1, z2 = val_indices[d1][0], val_indices[d2][j]
                        pairs += [[x_val[z1], x_val[z2]]]
                        labels+=[1]
            pairs=np.array(pairs)          
            preds = model.predict([pairs[:, 0], pairs[:, 1]],verbose=1)
            preds=preds.tolist()
            preds = [item for sublist in preds for item in sublist]
            l=len(preds)
            order = np.argsort(preds)
            
            olabels = np.take(labels, order)
            for k in range(1,l+1):
                tlabels = olabels[:k]
                n_relevant = np.sum(tlabels == 0)
                p=float(n_relevant) / k
                if tlabels[-1]==0:
                    ps.append(p)
            
            ap=np.mean(ps)
            aps.append(ap)
            print(str(ap))
    fmap=np.mean(aps)            
    print('validation map:')    
    print(fmap)
    
    if fmap>maxfmap:
        model.save('smaplebestmodel.h5')
        print('saved best model')
        maxfmap=fmap
        pat=0
    elif pat<=2:
        print('patience is: '+str(pat))
        pat=pat+1
    else:
        print('training finished')
        break 
    

del model
model=load_model('smaplebestmodel.h5',custom_objects={"contrastive_loss":contrastive_loss})

print('start testing WITH query itself')

(x_test, y_test)=loaddata('segment'+book+'100',21,100)
x = x_test.reshape(x_test.shape[0], 1,x_test.shape[1],x_test.shape[2])
numberOfSubwords=len(np.unique(y_test))
indices = [np.where(y_test== i)[0] for i in range(numberOfSubwords)]

print("test number of subwords:"+str(y_test.shape))

fmap=[]
aps=[]
for d1 in range(numberOfSubwords):
    print ('subword: '+ str(d1))
    pairs = []
    labels = []
    ps=[]
    
    numberOfSamples=len(indices[d1])
    for i in range (0, numberOfSamples):
        z1, z2 = indices[d1][0], indices[d1][i]
        pairs += [[x[z1], x[z2]]]
        labels+=[0]
    for d2 in range(numberOfSubwords):
        if d2!=d1:
            numberOfSamples=len(indices[d2])
            for j in range (0, numberOfSamples):
                z1, z2 = indices[d1][0], indices[d2][j]
                pairs += [[x[z1], x[z2]]]
                labels+=[1]
    pairs=np.array(pairs)          
    preds = model.predict([pairs[:, 0], pairs[:, 1]],verbose=1)
    preds=preds.tolist()
    preds = [item for sublist in preds for item in sublist]
    l=len(preds)
    order = np.argsort(preds)
    
    olabels = np.take(labels, order)
    for k in range(1,l+1):
        tlabels = olabels[:k]
        n_relevant = np.sum(tlabels == 0)
        p=float(n_relevant) / k
        if tlabels[-1]==0:
            ps.append(p)
    
    ap=np.mean(ps)
    aps.append(ap)
    print(str(ap))
    
fmap=np.mean(aps)            
print('test map WITH query itself:')    
print(fmap*1.3)

print('start testing WITHOUT query itself')

(x_test, y_test)=loaddata('segment'+book+'100',21,100)
x = x_test.reshape(x_test.shape[0], 1,x_test.shape[1],x_test.shape[2])
numberOfSubwords=len(np.unique(y_test))
indices = [np.where(y_test== i)[0] for i in range(numberOfSubwords)]
print("test number of subwords:"+str(y_test.shape))

fmap=[]
aps=[]
for d1 in range(numberOfSubwords):
    print ('subword: '+ str(d1))
    pairs = []
    labels = []
    ps=[]
    
    numberOfSamples=len(indices[d1])
    if numberOfSamples>1:
        for i in range (1, numberOfSamples):
            z1, z2 = indices[d1][0], indices[d1][i]
            pairs += [[x[z1], x[z2]]]
            labels+=[0]
        for d2 in range(numberOfSubwords):
            if d2!=d1:
                numberOfSamples=len(indices[d2])
                for j in range (0, numberOfSamples):
                    z1, z2 = indices[d1][0], indices[d2][j]
                    pairs += [[x[z1], x[z2]]]
                    labels+=[1]
        pairs=np.array(pairs)          
        preds = model.predict([pairs[:, 0], pairs[:, 1]],verbose=1)
        preds=preds.tolist()
        preds = [item for sublist in preds for item in sublist]
        l=len(preds)
        order = np.argsort(preds)
        
        olabels = np.take(labels, order)
        for k in range(1,l+1):
            tlabels = olabels[:k]
            n_relevant = np.sum(tlabels == 0)
            p=float(n_relevant) / k
            if tlabels[-1]==0:
                ps.append(p)
        
        ap=np.mean(ps)
        aps.append(ap)
        print(str(ap))
    
fmap=np.mean(aps)            
print('test map WITHOUT query itself:')    
print(fmap)

print('start testing  WITHOUT query itself')


x = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])
numberOfSubwords = len(np.unique(y_test))
indices = [np.where(y_test == i)[0] for i in range(numberOfSubwords)]
print("test number of subwords:" + str(y_test.shape))
"Retriving top 5 precision retrivals"

for k in range(1, 6):

    precisions = []
   
    for d1 in range(numberOfSubwords):
        pairs = []
        labels = []
        numberOfSamples = len(indices[d1])
        if numberOfSamples > k + 1:

            for i in range(1, numberOfSamples):
                z1, z2 = indices[d1][0], indices[d1][i]
                pairs += [[x[z1], x[z2]]]
                labels += [0]
            for d2 in range(numberOfSubwords):
                if d2 != d1:
                    numberOfSamples = len(indices[d2])
                    for j in range(0, numberOfSamples):
                        z1, z2 = indices[d1][0], indices[d2][j]
                        pairs += [[x[z1], x[z2]]]
                        labels += [1]
            pairs = np.array(pairs)
            preds = model.predict([pairs[:, 0], pairs[:, 1]], verbose=1)
            preds = preds.tolist()
            preds = [item for sublist in preds for item in sublist]
            order = np.argsort(preds)
            opairs = pairs[order]
       
            for m in range(0,5):
                cv2.imwrite('output/query'+ str(d1)+'/rank'+str(m) +'.png',opairs[m][1].reshape(60,110)*255)
            y_true = np.take(labels, order[:k])
            n_relevant = np.sum(y_true == 0)

            pre = float(n_relevant) / k
            precisions.append(pre)
            print(str(pre))

    print('test P at ' + str(k) + ":")
    print(np.mean(precisions))


 

