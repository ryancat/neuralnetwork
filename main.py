from numpy import exp, array, random, dot, arange
from matplotlib import pyplot
# import matplotlib
# import matplotlib.pyplot as plt

t = arange(0, 5, 0.2)

pyplot.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
pyplot.show()

class MyNeuralNetworkLayer():
    def __init__ (self, num_of_input, num_of_output):
        # random.seed(1)
        self.synaptic_weights = 2 * random.random((num_of_input, num_of_output)) - 1

    def get_weight_sum (self, inputs):
        return dot(inputs, self.synaptic_weights)

class MyNeuralNetwork():
    def __init__ (self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    def __sigmoid (self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_gradient (self, x):
        return x * (1 - x)

    def think (self, inputs):
        layer1_output = self.__sigmoid(self.layer1.get_weight_sum(inputs))
        layer2_output = self.__sigmoid(self.layer2.get_weight_sum(layer1_output))
        return layer1_output, layer2_output

    def train (self, train_input, train_output, num_of_iter):
        for i in range(num_of_iter):
            layer1_output, layer2_output = self.think(train_input)
            # print 'layer2_output'
            # print layer2_output
            # print '--------------------'

            layer2_error = train_output - layer2_output
            layer2_delta = layer2_error * self.__sigmoid_gradient(layer2_output)
            # print 'layer2_error'
            # print layer2_error
            # print layer2_delta
            # print '--------------------'

            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_gradient(layer1_output)
            
            # print 'layer1_error'
            # print layer1_error
            # print layer1_delta
            # print '--------------------'

            layer2_adjustment = layer1_output.T.dot(layer2_delta)
            layer1_adjustment = train_input.T.dot(layer1_delta)

            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

            # adjustment = dot(
            #     train_input.T, 
            #     error * self.__sigmoidGradient(self.__getWeightSum(train_input)))
            #     # error)
            # # print 'adjusted error'
            # # print error * self.__sigmoidGradient(self.__getWeightSum(train_input))
            # # print error
            # # # print 'train_input.T'
            # # # print train_input.T
            # # print 'adjustment'
            # # print adjustment
            # # print '--------------------'

            # self.synaptic_weights += adjustment
            # # print 'synaptic_weights'
            # # print self.synaptic_weights
            # # print '--------------------'

if __name__ == '__main__':
    neuralNetwork = MyNeuralNetwork(MyNeuralNetworkLayer(3, 4), MyNeuralNetworkLayer(4, 1))
    print 'Random starting synaptic weights: '
    print neuralNetwork.layer1.synaptic_weights
    print neuralNetwork.layer2.synaptic_weights

    train_input = array([
        [ 0,0,1 ],
        [ 0,1,1 ],
        [ 1,0,1 ],
        [ 1,0,0 ],
        [ 0,1,0 ],
        [ 1,1,0 ],
        [ 1,1,1 ],
        [ 0,0,0 ]])
    train_output = array([[1,0,0,0,0,1,1,1]]).T

    print 'Train input: '
    print train_input

    print 'Train output: '
    print train_output

    neuralNetwork.train(train_input, train_output, 60000)

    # print 'New synaptic weights after training: '
    # print neuralNetwork.synaptic_weights

    print 'Start thinking...'
    test_input = [1, 0, 0]
    print 'Consider new input', test_input
    layer1_output, layer2_outout = neuralNetwork.think(array(test_input))
    print layer2_outout