from __future__ import print_function

import csv

import cv2
import tensorflow as tf

from keras.engine import Layer
from keras.preprocessing.image import load_img
from keras.utils import np_utils

import keras
from keras.datasets import mnist
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Flatten, Conv1D, MaxPooling1D, Dense, Activation, \
    Dropout, GlobalMaxPooling1D, AveragePooling2D, ConvLSTM2D, GlobalMaxPooling2D, Recurrent, Reshape, Bidirectional, \
    BatchNormalization, Merge, concatenate, Permute, activations
from keras.utils import  to_categorical
from keras.layers import Embedding, add
from pandas.core.frame import DataFrame

from keras.layers import MaxPooling1D
from keras.optimizers import Adam
import pandas as pd
import sklearn.model_selection
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from keras import regularizers, Model, Input
import random

from sklearn.model_selection import train_test_split,StratifiedShuffleSplit

from graph_convolution import GraphConv
from utils import generate_Q

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,  decay=3e-8)

def attention_horizontal(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim1 = int(inputs.shape[1])
    input_dim2 = int(inputs.shape[2])
    input_dim3 = int(inputs.shape[3])

    a = Permute((3, 1,2))(inputs)
    a = Reshape((input_dim3, input_dim2,input_dim1))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim2, activation='softmax')(a)

    a_probs = Permute((3,2,1))(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return a_probs


def attention_vertical(inputs2):
    input_dim1 = int(inputs2.shape[1])
    input_dim2 = int(inputs2.shape[2])
    input_dim3 = int(inputs2.shape[3])

    a = Permute((3, 2,1))(inputs2)
    a = Reshape((input_dim3, input_dim2,input_dim1 ))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim2, activation='softmax')(a)

    b_probs = Permute((3,2,1))(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return b_probs

class D_Att(Layer):

    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = activations.get(activation)
        super(D_Att, self).__init__(**kwargs)


    def call(self, x):
        # x is a list (Feature matrix, Laplacian (Adjcacency) Matrix).
        assert isinstance(x, list)

        X, A1,A2 = x
        image_size = X.get_shape()[2].value
        #
        # A11 = tf.reshape(A1,[-1,image_size,image_size,self.units,1])
        # A22 = tf.reshape(A2,[-1,image_size,image_size,self.units,1])
        #
        # Aall = K.concatenate([A11,A22])
        # Amax = K.argmax(Aall)
        # Amax = tf.reshape(Amax,[-1,image_size,image_size,self.units])
        #
        #
        # Amax = K.cast(Amax,dtype='float32')
        #
        # A1 = K.exp(A1)
        # A2 = K.exp(A2)

        A = tf.multiply(A1,A2)
        A = K.exp(A)


        Maximum = tf.maximum(A1,A2)   #miniimum
        concatenate1 = K.concatenate([A, Maximum], axis=3)
        concatenate2 = K.concatenate([concatenate1, X], axis=3)

        # concatenate3 = K.concatenate([concatenate2, Amax], axis=3)
        return concatenate2

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        image_size = input_shape[0][1]
        input_dim = (input_shape[0][0],image_size,image_size, 3 * self.units)
        return input_dim


num_neighbors1 = 24
num_neighbors2 = 24
num_neighbors3 = 6
num_neighbors4 = 3


q_mat_layer1 = generate_Q(8,4)
q_mat_layer1 = np.argsort(q_mat_layer1,1)[:,-num_neighbors1:]   #将其地址存入GrapthConv

q_mat_layer2 = generate_Q(6,4)
q_mat_layer2 = np.argsort(q_mat_layer2,1)[:,-num_neighbors2:]   #将其地址存入GrapthConv

q_mat_layer3 = generate_Q(4,4)
q_mat_layer3 = np.argsort(q_mat_layer3,1)[:,-num_neighbors3:]   #将其地址存入GrapthConv

q_mat_layer4 = generate_Q(2,4)
q_mat_layer4 = np.argsort(q_mat_layer4,1)[:,-num_neighbors4:]   #将其地址存入GrapthConv


new_path1 = r'C:\Users\Administrator\Desktop\3\indian\indian_结果标签_1.csv'
new_path2 = r'C:\Users\Administrator\Desktop\3\indian\indian_label_1.csv'

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,  decay=3e-8)


subtrainfeature1 = pd.read_csv(r'C:\Users\Administrator\Desktop\3\indian\indian_feature_1.csv')  #特征
subtrainLabel1 = pd.read_csv(r'C:\Users\Administrator\Desktop\3\indian\indian_label_1_101.csv')  #Ground Truth标签

subtrain = pd.merge(subtrainfeature1,subtrainLabel1,on='Id')
from sklearn.utils import shuffle
# subtrain = shuffle(subtrain)
labels = subtrain.Class
subtrain.drop(["Class","Id"], axis=1, inplace=True)
subtrain = subtrain.values
(x_train, x_test,y_train,y_test)=train_test_split(subtrain,labels,test_size=0.85,stratify=labels)

print(y_train)
new_path3 = r'C:\Users\Administrator\Desktop\3\indian\indian_train2.csv'
# f = open(new_path3,'w')
# csv_write = f.writer(y_train,dialect='excel')
y_train.to_csv(new_path3,index=False,header=False)


# x_train = subtrain[0:10000]
x_test = subtrain[0:]
print(x_test.shape)
# # y_train = labels[0:10000]
y_test = labels[0:]

# x_train = subtrain[0:10000]
# x_test = subtrain[0:]
# # y_train = labels[0:10000]
# y_test = labels[0:]


# y_train = keras.utils.to_categorical(y_train, num_classes=2)
# y_test = keras.utils.to_categorical(y_test, num_classes=2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

y_train=np_utils.to_categorical(y_train, num_classes=16)
y_test=np_utils.to_categorical(y_test, num_classes=16)






model1_7 = Input( shape=(100,))
x = Dense(units=256)(model1_7)
x = Reshape((16, 16,1))(x)
x = Conv2D(filters=96, kernel_size=5, activation='relu',strides=2, padding='same')(x)
xall = Conv2D(filters=128, kernel_size=3, activation='relu',strides=1, padding='same')(x)


x = Conv2D(filters=128, kernel_size=3, activation='relu',strides=2, padding='same')(xall)
att_x = attention_horizontal(x)
att_x2 = attention_vertical(x)
x = D_Att(128)([x, att_x,att_x2])
x = Conv2D(filters=128, kernel_size=1, activation='relu',strides=1, padding='same')(x)
x = AveragePooling2D(2,strides=1, padding='same')(x)
#
att_x = attention_horizontal(x)
att_x2 = attention_vertical(x)
x = D_Att(128)([x, att_x,att_x2])
x = Conv2D(filters=128, kernel_size=1, activation='relu',strides=1, padding='same')(x)
x1 = AveragePooling2D(2,strides=1, padding='same')(x)




x = Conv2D(filters=128, kernel_size=3, activation='relu',strides=2, padding='same')(xall)
x = Reshape((64,32))(x)
x = GraphConv(filters=128, neighbors_ix_mat = q_mat_layer1, num_neighbors=24, activation='relu')(x)
x = Reshape((8,8,128))(x)
x = Conv2D(filters=128, kernel_size=1, activation='relu',strides=1, padding='same')(x)
x = Conv2D(filters=128, kernel_size=3, activation='relu',strides=1, padding='same')(x)
x = AveragePooling2D(2, padding='same')(x)
x = Reshape((16,128))(x)
x = GraphConv(filters=128, neighbors_ix_mat = q_mat_layer3, num_neighbors=6, activation='relu')(x)
x = Reshape((4,4,128))(x)
x = Conv2D(filters=128, kernel_size=1, activation='relu',strides=1, padding='same')(x)

x2 = AveragePooling2D(2,strides=1, padding='same')(x)

x = concatenate([x1,x2])

# x = Conv2D(filters=256, kernel_size=3, activation='relu',strides=2, padding='same')(x)
# x = Conv2D(filters=256, kernel_size=2, activation='relu',strides=1)(x)

x = Flatten()(x)


x = Dense(256,activation='relu')(x)
x = Dropout(0.5)(x)

all1_output = Dense(16)(x)
all1_output = Activation('softmax')(all1_output)


model1 = Model(inputs=[model1_7], outputs=[all1_output])
model1.summary()
model1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# model1.fit(x=x_train,y=y_train,batch_size=500,nb_epoch=200,verbose=2,validation_data=(x_test,y_test))

# loss,acc = model1.evaluate(x_test,y_test,verbose=2)
# print(acc)



from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath="best_model.h5",
                             monitor='val_acc',
                             verbose=1,
                             save_best_only='True',
                             save_weights_only='True',
                             mode='max',#(如果监视器monitor选val_acc, mode就选'max',如果monitor选acc,mode也可以选'max',如果monitor选loss,mode就选'min'),一般情况下选'auto',
                        period=1)#(checkpoints之间间隔的epoch数)
#损失不下降，则自动降低学习率

lrreduce=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

import time
fit_start = time.clock()
history= model1.fit(x=x_train,y=y_train,batch_size= 1000,epochs=500,verbose=2,validation_data=(x_test,y_test),callbacks = [checkpoint])
fit_end = time.clock()

print("train time is: ",fit_end-fit_start)

model1.load_weights('best_model.h5')

t_start = time.clock()
loss,acc = model1.evaluate(x_test,y_test,verbose=2)
t_end = time.clock()
print('Test loss :',loss)
print('Test accuracy :',acc)
print("test time is: ",t_end-t_start)



y_pred_class = model1.predict(x_test)

data1 = DataFrame(y_pred_class)
data1.to_csv(r'C:\Users\Administrator\Desktop\3\indian\indian_结果标签.csv',index=False,header=False)


'----------------------------------------------------------------------------------------------------------------------'
list3 = []


lines = y_pred_class.tolist()



f=open(new_path1, mode='w')

a = 0
for line in lines:
    if line:
        a = a + 1
        # print(line.index(max(line)))
        f.write(str(line.index(max(line))))
        f.write('\n')
        list3.append(line.index(max(line)))
        # list.append('\n')
f.close()

list22 =[ ]
f2 = open(new_path2,mode='r')
list_true = f2.readlines()

for i in list_true:
    ii = i
    ii = ii.replace('\n','')
    ii = int(ii)
    list22.append(ii)

# print(list)

"----------------------------------------------------------------------------------------------------------------------"
all_path = r'C:\Users\Administrator\Desktop\3\indian'

path1 = all_path + '\indian_label.csv'
path2 = all_path + '\indian_结果标签_1.csv'
path3 = all_path + '\indian_label_zui.csv'



f1=open(path1, mode='r')
f2=open(path2, mode='r')
f3=open(path3, mode='w')

line1 = f1.readlines()
line2 = f2.readlines()

a = 0

for i1 in line1:
    if i1 != '16\n':
        i1 = line2[a]
        a = a + 1
        # print(a)
        f3.write(i1)
    else:
        f3.write(i1)
        continue

f3.close()
"----------------------------------------------------------------------------------------------------------------------"
import numpy as np
aa = 0
path3 = all_path + '\indian_label_zui.csv'
f3=open(path3, mode='r')
list = f3.readlines()
for i in range(len(list)):
    aa = aa + 1
    if list[i] == '1\n':
        list[i] =[255,255,102]
    elif list[i] == '2\n':
        list[i] =[0,48,205]
    elif list[i] == '3\n':
        list[i] = [255, 102, 0]
    elif list[i] == '4\n':
        list[i] =[0,255,154]
    elif list[i] == '5\n':
        list[i] =[255,48,205]
    elif list[i] == '6\n':
        list[i] =[102,0,255]
    elif list[i] == '7\n':
        list[i] =[0,154,255]
    elif list[i] == '8\n':
        list[i] =[0,255,0]
    elif list[i] == '9\n':
        list[i] =[129,129,0]
    elif list[i] == '10\n':
        list[i] = [129, 0, 129]
    elif list[i] == '11\n':
        list[i] = [48, 205, 205]
    elif list[i] == '12\n':
        list[i] = [0, 102, 102]
    elif list[i] == '13\n':
        list[i] = [48, 205, 48]
    elif list[i] == '14\n':
        list[i] = [102, 48, 0]
    elif list[i] == '15\n':
        list[i] = [102, 255, 255]
    elif list[i] == '0\n':
        list[i] = [255, 255, 0]
    else:
        list[i] = [0, 0, 0]
data = np.reshape(list, (145, 145, 3))
print(len(list))
import scipy.misc
scipy.misc.imsave(r'C:\Users\Administrator\Desktop\3\indian\CNN_indian_0.05_CAG.jpg', data)

print('11')





from sklearn import metrics


def kappa(confusion_matrix, k):
    dataMat = np.mat(confusion_matrix)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe  = float(ysum*xsum)/np.sum(dataMat)**2
    OA = float(P0/np.sum(dataMat)*1.0)
    cohens_coefficient = float((OA-Pe)/(1-Pe))
    return cohens_coefficient

classify_report = metrics.classification_report(list22, list3)
confusion_matrix = metrics.confusion_matrix(list22, list3)
overall_accuracy = metrics.accuracy_score(list22, list3)
acc_for_each_class = metrics.precision_score(list22, list3, average=None)
average_accuracy = np.mean(acc_for_each_class)
kappa_coefficient = kappa(confusion_matrix, 16)
print('classify_report : \n', classify_report)
print('confusion_matrix : \n', confusion_matrix)
print('acc_for_each_class : \n', acc_for_each_class)
print('average_accuracy: {0:f}'.format(average_accuracy))
print('overall_accuracy: {0:f}'.format(overall_accuracy))
print('kappa coefficient: {0:f}'.format(kappa_coefficient))

newpath = r'C:\Users\Administrator\Desktop\3\indian\CNN_indian_0.05_CAG.txt'
f = open(newpath,'w')
f.write(classify_report)
# f.write(confusion_matrix)
f.write(str(acc_for_each_class.tolist()))
f.write('\n')
f.write('average_accuracy:{0:f}'.format(average_accuracy))
f.write('\n')
f.write('overall_accuracy:{0:f}'.format(overall_accuracy))
f.write('\n')
f.write('kappa coefficient:{0:f}'.format(kappa_coefficient))
f.write('\n')

f.close()