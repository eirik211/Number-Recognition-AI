import numpy as np
import matplotlib.pyplot as plt
from time import process_time


""" sigmoid function """
def s(x):
    return 1 / (np.exp(-x) + 1)     


""" partial-derivative of the sigmoid function """
def ds(x):
    return s(x) * (1 - s(x))


class Model():
    
    def __init__(self, interimAccuracy=True):

        """ net-variables """
        self.n = [np.zeros((784, 1)), np.zeros((20, 1)), np.zeros((10, 1))]                             # neuron-matrices
        self.z = [np.zeros((784, 1)), np.zeros((20, 1)), np.zeros((10, 1))]                             # neuron-matrices (before the activation function)
        self.w = [np.random.uniform(-0.5, 0.5, (20, 784)), np.random.uniform(-0.5, 0.5, (10, 20))]      # weight-matrices --> initialized randomly
        self.b = [np.zeros((20, 1)), np.zeros((10, 1))]                                                 # bias-matrices

        """ learning configuration """
        self.lr = 0.01          # learning rate 
        self.eps = 3            # epochs
        self.testC = 10000      # how many pictures should be saved for testing

        """ MNIST dataset """
        self.images = None
        self.labels = None     

        """ time- and accuracy-measurements """
        self.tstart = process_time()
        self.corrects = 0
        self.interimAcc = interimAccuracy 
    
    
    def getMNIST(self):
        """ load the MNIST-dataset """
        with np.load("mnist.npz") as f:
            images, labels = f["x_train"], f["y_train"]                                                 # load data
            images = images / 255                                                                       # convert to 0 - 1 grayscale
            self.images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))      # reshape images to list of (784,1) matrices
            self.labels = np.eye(10)[labels]                                                            # reshape labels to list of (10,1) matrices
        
    
    """ forward-propagation (calculate output) """
    def forwardP(self, i):
        self.n[0] = self.images[i].reshape(784, 1)              # the images' pixel represent the input 
        self.n[1] = s(self.b[0] + self.w[0] @ self.n[0])        # calculate the hidden-layer
        self.n[2] = s(self.b[1] + self.w[1] @ self.n[1])        # calculate the output-layer
    

    """ back-propagation (adjust weights) """
    def backP(self, i):
        """ calculate ∂C/∂w[1], ∂C/∂b[1], ∂C/w[0] and ∂C/∂b[0] in order to adjust the weights properly """
        # note: z = n before the sigmoid function was applied
        DcDn = 2 * (self.n[2] - self.labels[i].reshape(10, 1))                 # ∂C/∂n[2] 
        DnDz = self.n[2] * (1 - self.n[2])                                     # ∂n[2]/∂z[2]
        DzDw = np.transpose(self.n[1])                                         # ∂z[2]/∂w[1]
        self.w[1] += -self.lr * (DcDn * DnDz @ DzDw)                           # w[1] += -lr(∂C/∂w[1])
        self.b[1] += -self.lr * (DcDn * DnDz)                                  # b[1] += -lr(∂C/∂b[1])
        
        DcDn = np.transpose(self.w[1]) @ (DcDn * DnDz)                         # ∂C/∂n[1] = (∂C/∂n[2])(∂n[2]/∂z[2])(∂z[2]/n[1]) --> sum over the whole layer
        DnDz = self.n[1] * (1 - self.n[1])                                     # ∂n[1]/∂z[1]
        DzDw = np.transpose(self.images[i].reshape(784, 1))                    # ∂z[1]/∂w[0]
        self.w[0] += -self.lr * DcDn * DnDz @ DzDw                             # w[0] += -lr(∂C/∂w[0])
        self.b[0] += -self.lr * DcDn * DnDz                                    # b[0] += -lr(∂C/∂b[0])


    """ train the model """
    def train(self):
        self.getMNIST()
        
        for ep in range(self.eps):
            self.corrects = 0                                                    
            for i in range(len(self.images) - self.testC):
                self.forwardP(i)
                self.backP(i)

                """ save the correctness and print out the interim accuracy """
                if self.labels[i].argmax() == self.n[2].argmax():
                    self.corrects += 1
                if self.interimAcc:
                    if i%int((60000-self.testC)/10) == 0 and i != 0:
                        print(f" ----- {ep+1}. Epoch --> Finished: {i // ((60000-self.testC)/10) * 10}%. Accuracy: {round(self.corrects/(i+1) * 100, 3)}%")
            print(f"Finished the {ep+1}th epoch with an accuracy of {round(self.corrects/(i+1) * 100, 3)}%.")

        print(f"Finished training in {round(process_time() - self.tstart, 1)} seconds.", end="\n\n")


    """ test the model with unseen images """
    def test(self):
        self.corrects = 0
        for i in range(self.testC):
            index = 60000 - self.testC + i
            self.forwardP(index)
            if self.labels[index].argmax() == self.n[2].argmax():
                self.corrects += 1
        print(f"Accuracy for new images ({self.testC}): {round(self.corrects/self.testC * 100, 3)}%\n")
        
                
    """ let the user see the model in action """
    def testMode(self):
        uin = input("test mode? (y/n): ")
        while True:
            if uin == "y":
                index = input(f"Enter a number or \"x\" to exit (1 - {self.testC}): ")

                if index == "x":
                    return 0

                if not index.isnumeric() or int(index) < 1 or int(index) > self.testC:
                    print("invalid input")
                    break

                index = 60000 - self.testC + int(index) - 1

                self.forwardP(index)

                plt.imshow(self.images[index].reshape(28, 28), cmap="Greys")
                plt.title(f"It's probably a {self.n[2].argmax()}.\n")
                plt.show()
            elif uin == "n":
                return 0
            else:
                print("invalid input")
                self.testMode()
                return 0


model = Model(interimAccuracy=False)
model.train()
model.test()
model.testMode()

