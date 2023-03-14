import numpy as np


class Perceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size


        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):

        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)


        return self.a2

    def backward(self, X, y, y_pred, learning_rate):

        delta2 = (y_pred - y) * self.sigmoid_derivative(y_pred)
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)


        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def fit(self, X, y, epochs=1000, learning_rate=0.1):
        for i in range(epochs):

            y_pred = self.forward(X)

            error = np.mean(np.abs(y - y_pred))


            self.backward(X, y, y_pred, learning_rate)

    def predict(self, X):

        return self.forward(X)
    def inference(self, X):
        pred=self.forward(X)
        pred=np.round(pred)
        return pred
if __name__=="__main__" :

    X = np.array([[0,0,0], [0,0, 1], [0,1,0],[0,1,1][1,0, 0], [1, 0,1],[1,1,0],[1,1,1]])
    y = np.array([[0], [1], [1], [0],[1],[0],[0],[1]])


    perceptron = Perceptron(3, 4, 1)


    perceptron.fit(X, y)


    y_pred = perceptron.inference(X)
    print(y_pred)
    print(perceptron.W1)
    print(perceptron.W2)