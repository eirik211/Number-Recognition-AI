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
![](https://latex.codecogs.com/svg.image?\color{white}&space;\frac{\partial&space;zo}{\partial&space;wx}&space;=&space;nh)<br/>

Putting everything together...

![](https://latex.codecogs.com/svg.image?\color{white}&space;\frac{\partial&space;C}{\partial&space;wx}&space;=&space;2(no&space;-&space;desiredOutput)sigmoid'(zo)nh)

The next step is to multiply the just calculated value with a learning rate `lr = 0.01` and subtract it from `wx`. <br/>
We use this learning rate to prevent the net from learning too fast, because it leads to bad results.


![](https://latex.codecogs.com/svg.image?\color{white}wx&space;=&space;wx&space;-&space;lr&space;*&space;2(no&space;-&space;desiredOutput)sigmoid'(zo)nh)

This proccess is then repeated for every other weight, that is connecting the hidden- with the output-layer.

### Adjusting the weights between the input- and the hidden-layer
Let's once again look at just one weight and call it `wy`. It connects `ni` (neuron in the input layer) and `nh` (neuron in the hidden layer).
This procedure is very similar to the one above. The only difference is, that the partial derivative of the cost with respect to `nh` is not as easy to calculate. Since `nh` is connected to every neuron of the output-layer, we have to sum up the impact it has trough each weight-connection to the output layer.

Notations: o = output-layer, z = output-layer before sigmoid was applied <br/><br/>
![](https://latex.codecogs.com/svg.image?\color{white}&space;\frac{\partial&space;C}{\partial&space;nh}&space;=&space;&space;\sum_{i=0}^{9}&space;\frac{\partial&space;C}{\partial&space;o_{i}}&space;\frac{\partial&space;o_{i}}{\partial&space;z_{i}}&space;\frac{\partial&space;z_{i}}{\partial&space;nh}&space;)

Now we can just apply the chain rule like previously. <br/>
Note: `zh` = `nh` before sigmoid was applied <br/>

![](https://latex.codecogs.com/svg.image?\color{white}&space;\frac{\partial&space;C}{\partial&space;wy}&space;=&space;\frac{\partial&space;C}{\partial&space;nh}&space;\frac{\partial&space;nh}{\partial&space;zh}&space;\frac{\partial&space;zh}{\partial&space;wy})
<br/><br/>
![](https://latex.codecogs.com/svg.image?\color{white}&space;\frac{\partial&space;C}{\partial&space;wy}&space;=&space;(\sum_{i=0}^{9}\frac{\partial&space;C}{\partial&space;o_{i}}\frac{\partial&space;o_{i}}{\partial&space;z_{i}}\frac{\partial&space;z_{i}}{\partial&space;nh})sigmoid'(zh)ni)






