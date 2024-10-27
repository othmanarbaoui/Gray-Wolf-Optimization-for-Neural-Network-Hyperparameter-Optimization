import numpy as np
import random
from keras.datasets import mnist
from keras.utils import to_categorical

from neural_network import Layer , neural_network

class GWOptimazer:
    def __init__(self, max_iter):
        self.max_iter = max_iter 
        
    def calcul_fitness(self,position,x_train,y_train,x_test,y_test):

        net = neural_network()

        net.addLayer(Layer(28*28,position[1].astype(int)))
        for _ in range(position[0].astype(int)):
            net.addLayer(Layer(position[1].astype(int),position[1].astype(int)))
        net.addLayer(Layer(position[1].astype(int),10))


        net.fit(x_train[0:100],y_train[0:10000],10,position[2])
        
        predicted_value = np.array(net.predict(x_test[0:10]))
        predicted_value = predicted_value.astype('float32')
        true_values = np.array(y_test[0:10])
        return net.calculate_accuracy(predicted_value,true_values)
    
    def optimazer(self,x_train,y_train,x_test,y_test):
        iter = 0
        num_wolf = 4
        dimension = 3
        
        min_bound = [3,16,0.0001]
        max_bound = [9,256,0.1]
        
        position = np.random.uniform(min_bound,max_bound,size=(num_wolf,dimension))
        position_int = position
        fitness_value = np.zeros(num_wolf)
        
        for i in range(position_int.shape[0]):
            fitness_value[i] = self.calcul_fitness(position_int[i],x_train,y_train,x_test,y_test)  
        
        a = 2*(1-0/self.max_iter)
        A = 2*random.random()*a - a
        C = 2*random.random()
        
        while( iter < self.max_iter):
            
            fitness_value_sorted = np.sort(fitness_value)[::-1]
            
            alpha = np.where(fitness_value == fitness_value_sorted[0])[0][0]
            beta = np.where(fitness_value == fitness_value_sorted[1])[0][0]
            delta = np.where(fitness_value == fitness_value_sorted[2])[0][0]
        
            x_alpha = position_int[alpha]
            x_beta = position_int[beta]
            x_delta = position_int[delta]
            
            for wolf in range(num_wolf):
                
                x = position_int[wolf]
                
                D_alpha = abs(C*x_alpha-x)
                D_beta = abs(C*x_beta-x)
                D_delta = abs(C*x_delta-x)
                
                x1 = abs(x_alpha-A*D_alpha)
                x2 = abs(x_beta-A*D_beta)
                x3 = abs(x_delta-A*D_delta)
                
                new_x = ((x1+x2+x3)/3)
                new_x = new_x.reshape(1,dimension)
                
                if new_x[0, 0] > 12:
                    new_x[0, 0] = 12
                elif new_x[0, 0] < 1:
                    new_x[0, 0] = 1
                
                if new_x[0, 1] < 16:
                    new_x[0, 1] = 16
                elif new_x[0, 1] > 256:
                    new_x[0, 1] = 256
                
                new_fitness_value = self.calcul_fitness(new_x[0],x_train,y_train,x_test,y_test)
                
                if(new_fitness_value > fitness_value[wolf]):
                    position_int[wolf] = new_x
                    fitness_value[wolf] = new_fitness_value   
                                    
            iter = iter + 1
            
            a = 2*(1-0/self.max_iter)
            A = 2*random.random()*a - a
            C = 2*random.random()
            
        return x_alpha,fitness_value[alpha]

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255

y_train = to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)
        
GWO = GWOptimazer(3)
hyperparam,my_fitness_value = GWO.optimazer(x_train,y_train,x_test,y_test)
print('the best hyperparameters :')
print('hidden layer : ',hyperparam[0].astype(int))
print('every layer have : ', hyperparam[1].astype(int))
print('the learning rate : ',hyperparam[2])
print('the accuracy of this hyperparameters :', my_fitness_value)