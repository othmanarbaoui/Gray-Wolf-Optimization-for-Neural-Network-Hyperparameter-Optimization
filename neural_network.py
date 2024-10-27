import numpy as np
def sigmoid(x):
   return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return np.exp(-x) / (1 + np.exp(-x))**2

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


class Layer:
    def __init__(self,n_input,n_neuron):
        self.weigths = np.random.randn(n_input,n_neuron)*0.1
        self.bias = np.random.randn(1,n_neuron)*0.1
        
    def forward(self,input):
        self.input = input
        self.output = np.dot(self.input,self.weigths) + self.bias
        self.activate_output = sigmoid(self.output)
        return self.activate_output
    
    def backward(self, error, learning_rate):
        actiavted_input_error = sigmoid_prime(self.activate_output)*error
        input_error = np.dot(actiavted_input_error,self.weigths.T)
        weights_error = np.dot(self.input.T, error)
        bias_error = np.sum(error, axis=0, keepdims=True)
        
        self.weigths -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return input_error
        
        

class neural_network:
    def __init__(self):
        self.layers = []
        self.losse = mse
        self.losse_prime = mse_prime
        
    def addLayer(self, layer):
        self.layers.append(layer)
        
    
    def fit(self, x_train , y_train , epochs, learning_rate):
        samples = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)
                err+=self.losse(y_train[j],output)
                 
                error = self.losse_prime(y_train[j],output)
                for layer in reversed(self.layers):
                    error = layer.backward(error,learning_rate)
                
            print('____________ epoch = %d , error = %f ____________' % (i+1,err/samples))
            
            
    def predict(self,input_data):
        samples = len(input_data)
        result = []
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            max_index = np.argmax(output[0])  
            converted_output = [0] * len(output[0])  
            converted_output[max_index] = 1  
            result.append(converted_output)
            
        return result
            
    
    def calculate_accuracy(self,predicted_values, true_values):

        if predicted_values.shape[0] != true_values.shape[0]:
            raise ValueError("Lengths of predicted values and true values should match.")
        
        i=0
        correct_predictions = 0
        while i < len(true_values):
            if np.array_equal(true_values[i], predicted_values[i]) == True:
                correct_predictions+=1
            i+=1
        total_predictions = true_values.shape[0]
        accuracy = correct_predictions / total_predictions
        
        return accuracy
                

        
