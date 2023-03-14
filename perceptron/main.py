import numpy as np
class Perceptron :
    def __init__(self,input_size,lr=0.5,epoch=100):
        self.W=np.random.random(input_size+1)
        self.epoch=epoch
        self.lr=lr
    def activation_fn(self,x):
        if x>=0 :
            return 1
        else :
            return 0
    def predict(self,x):
        x=np.insert(x,x.size,1)
        z=self.W.T.dot(x)
        a=self.activation_fn(z)
        return a
    def fit(self,X,d):
        for _ in range(self.epoch) :
            for i in range(d.shape[0]) :

                x=X[i]
                y=self.predict(x)
                e=d[i]-y
                x=np.insert(x,x.size,1)
                self.W=self.W+self.lr*e*x
if __name__=="__main__" :
    X=np.array([[0,0],
                [0,1],
                [1,0],
                [1,1]])
    d=np.array([0,1,1,1])
    perceptron=Perceptron(input_size=2)
    perceptron.fit(X,d)
    print(perceptron.W)
    print(perceptron.predict(np.array([0,0])))
    print(perceptron.predict(np.array([0, 1])))
    print(perceptron.predict(np.array([1, 0])))
    print(perceptron.predict(np.array([1, 1])))

