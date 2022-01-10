import numpy as np
from scipy.special import logsumexp

class BaseLayer:
    '''Basic layout of the layers'''
    def __init__(self):
        pass
    def forward(self):
        pass
    def update_params(self):
        pass
    
class Dense(BaseLayer):    
    def __init__(self, in_features, out_features, learning_rate = 0.1):   
        ''' Initialize the Dense layer parameters '''
        self.W = np.random.rand(in_features, out_features)
        self.b = np.random.rand(out_features)
        self.lr = learning_rate 
           
    def forward(self, inputs):        
        ''' f(x) = W'x + b '''
        self.x = inputs
        return np.dot(self.W.T, self.x) + self.b
    
    def backward(self, gradient):
        ''' 
        Dense layer D produces takes input x and gives output f.
        Gradient, df/dx = df/dD * dD/dx 
        where df/dD is the output gradient and dD/dx is the input gradient
        or gradient = output_gradient * W'
        
        Similarly, df/dW and df/db can be calculated as,
        gradient_W = x' * output_gradient
        gradient_b = mean(output_gradient, axis = 1)
        '''
        grad = np.dot(gradient, self.W.T)
        
        grad_W = np.dot(self.x.T, gradient)
        grad_b = np.sum(gradient, axis=0)
        self.update_params(grad_W,grad_b)
        
        return grad
     
    def update_params(self, grad_W, grad_b):
        '''Update the Dense layer parameters using the calculated gradients'''
        self.W -= self.lr*grad_W
        self.b -= self.lr*grad_b
    
class Conv(BaseLayer):    
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):        
        self.num_filter = out_channels
        (self.m, self.n) = kernel_size
        self.filter = np.random.rand(out_channels, self.m, self.n)   # Generate number of filters of size mxn
        self.stride = stride
        self.padding = padding
    
    def get_filter_regions(self, padded_image):
        '''Get mxn regions at position (i,j) in the image to convulate with the filter''' 
        regions = []
        [row, column] = [padded_image.shape[0], padded_image.shape[1]]
        outsize = column - 1
        for i in range(0, row-outsize, self.stride):
            for j in range(column-outsize):
                region = padded_image[i:i+self.m, j:j+self.n]
            regions.append(region)    
        return regions       
    def forward(self, inputs):        
        #padded_image = np.pad(inputs, padding)
        #rows, columns = padded_image.shape[0], padded
        return
    def backward(self, gradient):
        return
    
class AvgPool(BaseLayer):    
    def __init__(self, in_channels, kernel_size, padding, stride):        
        return    
    def forward(self, inputs):
        return
    def backward(self, gradient):
        return

class MaxPool(BaseLayer):    
    def __init__(self, in_channels, kernel_size, padding, stride):        
        return    
    def forward(self, inputs):
        return
    def backward(self, gradient):
        return
    
class Flatten(BaseLayer):
    def forward(self, inputs):
        self.original_shape = inputs.shape
        return np.ravel(inputs)
        
    def backward(self, gradient):        
        return gradient.reshape(self.original_shape)
    
class Dropout(BaseLayer):    
    def __init__(self, num):        
        return    
    def forward(self, ):
        return
    def backward(self, ):
        return
    
class Sigmoid(BaseLayer): 
    def forward(self, inputs):
        ''' 
        Implement the sigmoid activation function
        sig (x) = 1/(1-exp(-x))
        '''
        self.sigma = 1 / (1 - np.exp(-inputs))
        return self.sigma
    def backward(self, gradient):
        '''
        d sig(x)/dx = sig(x)*(1-sig(x))
        '''
        sig = self.sigma
        return gradient * sig * (1 - sig)
    
class Softmax(BaseLayer): 
        
    def forward(self, inputs):
        self.shape = inputs.shape
        np.exp(inputs - logsumexp(inputs, axis=-1, keepdims=True))
        self.output = inputs
        return inputs
    def backward(self, gradient):
        grad = gradient*self.output
        grad -= grad.sum()*self.output
        return grad
    
class ReLU(BaseLayer):
    def forward(self, inputs):
        '''Returns the input if it is > 0, else returns 0'''
        self.inp = inputs
        return np.maximum(0, inputs)
    
    def backward(self, gradient):
        '''If input > 0, return the gradient else 0'''
        return gradient*(self.inp > 0)

class MSE(BaseLayer):
    def __init__(self, predicted, actual):
        self.predicted = predicted
        self.actual = actual
             
    def forward(self):
        return np.power(self.predicted - self.actual, 2).mean()
 
    def backward(self):
        return 2 * (self.predicted - self.actual).mean()