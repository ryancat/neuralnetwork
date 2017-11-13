from numpy import exp, array, random, dot, arange
from matplotlib import pyplot
# import matplotlib
# import matplotlib.pyplot as plt

# t = arange(0, 5, 0.2)

# pyplot.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# pyplot.show()

class MyNeuralNetworkLayer():
    def __init__ (self, num_of_input, num_of_output):
        # random.seed(1)
        self.synaptic_weights = 2 * random.random((num_of_input, num_of_output)) - 1

    def get_weight_sum (self, inputs):
        return dot(inputs, self.synaptic_weights)

class MyNeuralNetwork():
    def __init__ (self, layers):
        self.layers = layers
            
    def __sigmoid (self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_gradient (self, x):
        return x * (1 - x)

    def think (self, inputs):
        for index, layer in enumerate(self.layers):
            layer.input = self.layers[index - 1].output if index > 0 else inputs
            layer.output = self.__sigmoid(layer.get_weight_sum(layer.input))

    def train (self, train_input, train_output, num_of_iter):
        for i in range(num_of_iter):
            self.think(train_input)
            for index, layer in reversed(list(enumerate(self.layers))):
                if index == len(self.layers) - 1:
                    layer.error = train_output - layer.output
                else:
                    next_layer = self.layers[index + 1]
                    layer.error = next_layer.delta.dot(next_layer.synaptic_weights.T)
                layer.delta = layer.error * self.__sigmoid_gradient(layer.output)
                layer.adjustment = layer.input.T.dot(layer.delta)
                layer.synaptic_weights += layer.adjustment

if __name__ == '__main__':
    # layers = [MyNeuralNetworkLayer(3, 4), MyNeuralNetworkLayer(4, 1)]
    layers = [MyNeuralNetworkLayer(3, 1)]
    neuralNetwork = MyNeuralNetwork(layers)
    print 'Random starting synaptic weights: '
    for layer in neuralNetwork.layers:
        print layer.synaptic_weights

    train_input = array([
        [ 0,0,1 ],
        [ 1,1,1 ],
        [ 1,0,1 ],
        [ 0,1,1 ]])
    train_output = array([[0, 1, 1, 0]]).T

    print 'Train input: '
    print train_input

    print 'Train output: '
    print train_output

    neuralNetwork.train(train_input, train_output, 10000)

    print 'Start thinking...'
    test_input = [1, 0, 0]
    print 'Consider new input', test_input
    neuralNetwork.think(array(test_input))
    print neuralNetwork.layers[-1].output