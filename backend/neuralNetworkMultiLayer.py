import numpy as np

class NeuronalNetwork:
    def __init__(self,inputs,output, layers):
        self.inputs                 = inputs
        self.output                 = output
        self.inputsLength           = len(inputs[0])
        self.outputLength           = len(output[0])
        self.layers                 = layers
        self.weight_output_layer    = np.random.rand(self.layers[-1], self.outputLength)
        self.lr                     = 0.1

        self.setWeightHiddenLayers()
        

    def sigmoid(self,x):
        return 1.0/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)   

    def setWeightHiddenLayers(self):
        self.weight_hidden_layers = []
        self.weight_hidden_layers.append(np.random.rand(self.inputsLength, self.layers[0]))
        for i in range(0,len(self.layers)-1):
            self.weight_hidden_layers.append(np.random.rand(self.layers[i],self.layers[i+1]))

    def acumulateActivationHiddenLayers(self):
        self.layers_results = []
        self.layers_results.append(self.sigmoid(np.dot(self.inputs, self.weight_hidden_layers[0]))) 
        for idx in range(0,len(self.weight_hidden_layers) - 1):
            self.layers_results.append(self.sigmoid(np.dot( self.layers_results[idx], self.weight_hidden_layers[idx+1]))) 

        return self.layers_results[-1]

    def feedforward(self):
        self.hidden_layers_result   = self.acumulateActivationHiddenLayers()
        self.output_layer_result    = self.sigmoid(np.dot(self.hidden_layers_result, self.weight_output_layer))

    def backPropagation(self):
        # Error de la capa de salida
        self.predicted_error_output = 2 * (self.output - self.output_layer_result) * self.sigmoid_derivative(self.output_layer_result)
        self.error_hidden_layers    = []
        error_hidden_layer          = self.predicted_error_output.dot(self.weight_output_layer.T)

        self.error_hidden_layers.append(2 * error_hidden_layer * self.sigmoid_derivative(self.layers_results[-1]))

        for i in reversed(range(0,len(self.layers_results) -1 )):
            idx                 = len(self.error_hidden_layers) -1
            error_hidden_layer  = self.error_hidden_layers[idx].dot(self.weight_hidden_layers[i+1].T)
            self.error_hidden_layers.append(2 * error_hidden_layer * self.sigmoid_derivative(self.layers_results[i]))

        self.error_hidden_layers.reverse()
        

    def updateWeight(self):
        self.weight_output_layer        += self.hidden_layers_result.T.dot(self.predicted_error_output) * self.lr
        self.weight_hidden_layers[0]    += self.inputs.T.dot(self.error_hidden_layers[0]) * self.lr

        for i in range(0,len(self.weight_hidden_layers) -1):
            self.weight_hidden_layers[i+1] += self.layers_results[i].T.dot(self.error_hidden_layers[i+1]) * self.lr
        

    def train(self,epochs):
        for _ in range(epochs):
            self.feedforward()
            self.backPropagation()
            self.updateWeight()      

    def predict(self, input_vector):

        layerActivation =  self.sigmoid(np.dot(input_vector,self.weight_hidden_layers[0]))

        for i in range(1, len(self.weight_hidden_layers)):
            lastLayerActivation =  self.sigmoid(np.dot(layerActivation,self.weight_hidden_layers[i]))
            layerActivation = lastLayerActivation

        prediction = self.sigmoid(np.dot(layerActivation, self.weight_output_layer))
        return prediction[0]
