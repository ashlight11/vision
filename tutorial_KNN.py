import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt
from PIL import Image

basedir_data = "./data/"
rel_path = basedir_data + "cifar-10-batches-py/"

#Désérialiser les fichiers image afin de permettre l’accès aux données et aux labels:

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict
 

X = unpickle(rel_path + 'data_batch_1')
img_data = X[b'data']
img_label_orig = img_label = X[b'labels']
img_label = np.array(img_label).reshape(-1, 1)

'''
print('head of training set : ', img_data)
print('shape of training set : ', img_data.shape)
print('head of label of training set : ', img_label)
print('shape of label of training set : ', img_label.shape) '''

test_X = unpickle(rel_path + 'test_batch');
test_data = test_X[b'data']
test_label = test_X[b'labels']
test_label = np.array(test_label).reshape(-1, 1)

""" print('head of testing set : ', test_data)
print('shape of testing set : ', test_data.shape)
print('head of label of testing set : ', test_label)
print('shape of label of testing set : ', test_label.shape) """

meta = unpickle(rel_path + 'batches.meta')
meta = meta[b'label_names']
print(meta)

sample_img_data = img_data[0:10, :]
'''print(sample_img_data)
print('shape sample ', sample_img_data.shape)
print('shape sample', sample_img_data[1,:].shape)'''

'''one_img=sample_img_data[0,:]
r = one_img[:1024].reshape(32, 32)
g = one_img[1024:2048].reshape(32, 32)
b = one_img[2048:].reshape(32, 32)
rgb = np.dstack([r, g, b])
cv2.imshow('Image CIFAR',rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()'''


def pred_label_fn(i):
    return meta[YPred[i]].decode('utf-8')

from datetime import datetime

start = datetime.now()
nbrs = KNeighborsClassifier(n_neighbors=3, algorithm='brute').fit(img_data, img_label_orig)
duration = start - datetime.now()
duration = start.strftime("%H:%M:%S")
print("Duration =", duration)

# test sur les 10 premières images
data_point_no = 10
sample_test_data = test_data[:data_point_no, :]
sample_test_label = test_label[:data_point_no]

YPred = nbrs.predict(sample_test_data)
print(YPred )

def show_img(data, labels, index): 

    font = cv2.FONT_HERSHEY_SIMPLEX
    color_info = (255, 255, 255)

    one_img=data[index,:]
    r = one_img[:1024].reshape(32, 32)
    g = one_img[1024:2048].reshape(32, 32)
    b = one_img[2048:].reshape(32, 32)
    rgb = np.dstack([r, g, b])

    text =pred_label_fn(index)
    print(text)

    cv2.putText(rgb, text ,(10, 30), font, 0.2, color_info, 1, cv2.LINE_AA)
    cv2.imshow('Image CIFAR' ,rgb)
    

for i in range(0, len(YPred)):
    show_img(sample_test_data, sample_test_label, i)
    cv2.waitKey() 
    

cv2.destroyAllWindows()    