## Neural Network
The neural network consists of 3 layers:
+ input-layer: 784 neurons  --> pixels of the 28\*28 image
+ hidden-layer: 24 neurons  
+ output-layer: 10 neurons  --> matching with the numbers (0 - 9) <br/>
As activation function, sigmoid is used.

### Backpropagation
I use a proccess called backpropagation to adjust the weights.

After an images has gone trough the neural net, the output is very likely to be completely false, since the weights are initialized randomly.
To set give this falseness a value we use the cost function.

![Cost function](https://latex.codecogs.com/svg.image?\color{white}&space;C(...)&space;=&space;\sum_{n=0}^{9}(output[n]&space;-&space;desiredOutput[n])^{2})

Now we need to find out how sensitive the cost is to small changes to each weight, in order to adjust it.
In other words: find the partial-derivative of the cost function in respect to the weight.

This can easily be done by applying the chain rule.<br/>
Notations: <br/> 
+ n = neuron <br/> 
+ z = n before the activation function <br/> 
+ w = weight <br/>
+ x = index of the neuron which the weight is connected to
<br/>

![](https://latex.codecogs.com/svg.image?\color{white}\frac{\partial&space;C}{\partial&space;w}&space;=&space;\frac{\partial&space;C}{\partial&space;n}\frac{\partial&space;n}{\partial&space;z}\frac{\partial&space;z}{\partial&space;w})

Now we have 3 simpler equations to solve. <br/>

![](https://latex.codecogs.com/svg.image?\color{white}\frac{\partial&space;C}{\partial&space;n}&space;=&space;2(output[x]&space;-&space;desiredOutput[x]))
![](https://latex.codecogs.com/svg.image?\color{white}\frac{\partial&space;n}{\partial&space;z}&space;=&space;sigmoid'(z))
