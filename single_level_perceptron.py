from numpy import exp,array,random, dot,squeeze
import matplotlib.pyplot as plt

class neuralNetwork():
    def __init__(self):
        random.seed(42) #seeding to get a constant random value every time
        self.weights=(2 * random.random((3,1)) - 1)


    # the activation function
    def sigmoid(self,x):
        return 1/(1+exp(-x))

    # derivative of the sigmoid function. used for gradient descent
    def diff_sigmoid(self,x):
        return(x*(1-x))

    # to predict a value in the feed forward neural network
    def prediction(self,input):
        return(self.sigmoid(dot(input,self.weights)))


    #the weights are updated here
    def train(self,trainingDataset,result,epoch):
        err=[]

        for itt in range(epoch):
            out=self.prediction(trainingDataset) #the result generated from training
            error=result-out
            err.append(error)
            adj = dot(trainingDataset.T, error * self.diff_sigmoid(out))
            self.weights +=adj


        plot_err=squeeze(err) #helpful for ploting arrays
        #plot for individual error values
        plt.title("error")
        plt.plot(plot_err)
        plt.show()


if __name__=="__main__":
    neu=neuralNetwork()
    print(neu.weights)
    trainingDataset = array([[0, 0,0], [0,1, 0], [0, 0,1], [1,1,1],[1,0,0],[1,1,0]])#the input trainig dataset
    result = array([[0,1,0,1,0,1]]).T #the actual expected result
    neu.train(trainingDataset, result, 10)
    print(neu.weights)

    print("prediction",neu.prediction(array([0,1,0])))







