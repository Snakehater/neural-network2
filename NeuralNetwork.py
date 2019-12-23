# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
import numpy as np
import timeit

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def think(self, input):
        layer1 = sigmoid(np.dot(np.array(input), self.weights1))
        return sigmoid(np.dot(layer1, self.weights2))

    def train(self, iterations):
        self.printProgressBar(0, iterations, prefix = ' Progress:', suffix = 'Complete', length = 50)
        for i in range(iterations):
            self.feedforward()
            self.backprop()
            self.printProgressBar(i, iterations, prefix = ' Progress:', suffix = 'Complete', length = 50)
        self.printProgressBar(iterations, iterations, prefix = ' Progress:', suffix = 'Complete', length = 50)
    # Print iterations progress
    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()



if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [0,1,0],
                  [1,0,0],
                  [0,0,0]])
    y = np.array([[0],[1],[1],[0],[0],[0]])
    nn = NeuralNetwork(X,y)

    nn.train(100000)

    print(nn.output)
    print('\n')
    print(nn.weights1)
    print('\n')
    print(nn.weights2)
    print('\n')
    while True:
        print('\n')
        inputss = input('Input: ')
        usrInput1 = int(inputss[0])
        usrInput2 = int(inputss[1])
        usrInput3 = int(inputss[2])
        calculated = nn.think(np.array([usrInput1, usrInput2, usrInput3]))
        print(calculated)
        # print(int(round(calculated[0])))
