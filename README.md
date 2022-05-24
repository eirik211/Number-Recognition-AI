## Neural Network
The neural network consists of 3 layers:
+ input-layer: 784 neurons  --> pixels of the 28\*28 image
+ hidden-layer: 24 neurons  
+ output-layer: 10 neurons  --> matching with the numbers (0 - 9)
<br/> 
As activation function, sigmoid is used.

## Backpropagation
I use a proccess called backpropagation to adjust the weights.

After the first images has gone trough the neural net, the output is very likely to be completely false, since the weights were initialized randomly.
To set give this falseness a value we use the cost function.

![Cost function](https://latex.codecogs.com/svg.image?\color{white}&space;C(...)&space;=&space;\sum_{n=0}^{9}(output[n]&space;-&space;desiredOutput[n])^{2})

### Adjusting the weights between the hidden- and the output-layer

For the sake of simplicity, let's first look at **one** weight and call it `wx`. It creates a connection between `nh` (neuron somewhere in the hidden layer) and `no` (neuron somewhere in the output layer).<br/>

Now we need to find out how sensitive the cost is to small changes to `wx`, in order to adjust it. <br/>
In other words: find the partial-derivative of the cost function with respect to `wx`.<br/>

![](https://latex.codecogs.com/svg.image?\color{white}&space;\frac{\partial&space;C}{\partial&space;wx}&space;&space;=&space;?)
 
This can easily be done by applying the chain rule. Note: `zo` = `no` before the sigmoid function was applied.<br/>

![](https://latex.codecogs.com/svg.image?\color{white}&space;\frac{\partial&space;C}{\partial&space;wx}&space;=&space;\frac{\partial&space;C}{\partial&space;no}&space;\frac{\partial&space;no}{\partial&space;zo}&space;\frac{\partial&space;zo}{\partial&space;nh})

![](https://latex.codecogs.com/svg.image?\color{white}&space;\frac{\partial&space;C}{\partial&space;no}&space;=&space;2(no&space;-&space;desiredOutput))<br/><br/>
![](https://latex.codecogs.com/svg.image?\color{white}&space;\frac{\partial&space;no}{\partial&space;zo}&space;=&space;sigmoid'(zo))<br/><br/>
![](https://latex.codecogs.com/svg.image?\color{white}&space;\frac{\partial&space;zo}{\partial&space;wx}&space;=&space;nh)<br/><br/>

Putting everything together...

![](https://latex.codecogs.com/svg.image?\color{white}&space;\frac{\partial&space;C}{\partial&space;wx}&space;=&space;2(no&space;-&space;desiredOutput)sigmoid'(zo)nh)









