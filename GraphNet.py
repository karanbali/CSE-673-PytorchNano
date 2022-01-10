class Net:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            print(output.shape, layer)
            output = layer.forward(output)             
        return output
    
    def backward(self, gradient):
        back_grad = gradient
        reverse = self.layers.copy()
        reverse.reverse()        
        for layer in reverse:
            print(gradient.shape, layer)
            back_grad = layer.backward(back_grad)
        return back_grad