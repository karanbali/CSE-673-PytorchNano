from GraphNet import Net
from Layers import *

def create_model():
	Model = Net()
	Model.add(Flatten())
	Model.add(Dense(784,512))
	Model.add(ReLU())
	Model.add(Dense(512,256))
	Model.add(ReLU())
	Model.add(Dense(256,10))
	Model.add(Softmax())
	return Model
