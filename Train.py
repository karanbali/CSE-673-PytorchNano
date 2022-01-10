from Model import create_model

from tensorflow.keras.datasets.mnist import load_data

import numpy as np
from Layers import *

(x_train, y_train), (x_test, y_test) = load_data()
#---------------------------------------------
# The following method would create the model 
#---------------------------------------------
model = create_model()

epochs = 100
losses = []
loss_function = MSE
lr = 0.1

for epoch in range(epochs):
    print('Epoch {}'.format(epoch+1))        
    
    for i, (image, label) in enumerate(zip(x_train, y_train)):
               
        prediction = model.forward(image)
        classes = prediction.shape[0]
        one_hot = [0 for i in range(classes)]
        one_hot[label]=1
        loss = loss_function(prediction, label)        
        error = loss.forward()
        print(error)
        error_gradient = loss.backward()
        
        losses.append(error)
        
        gradient = model.backward(error_gradient)
        
        if i%10==0:
            print('Loss for step {} :{}'.format(i+1,loss))