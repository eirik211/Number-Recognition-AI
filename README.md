## Neural Network
The neural network consists of 3 layers:
+ input-layer: 784 neurons  --> pixels of the 28\*28 image
+ hidden-layer: 24 neurons  
+ output-layer: 10 neurons  --> matching with the numbers (0 - 9)
<br/> 
As activation function, sigmoid is used.

## Backpropagation
Notations: <br/> 
+ n = neuron<br/> 
+ z = n before the activation function <br/> 
+ w = weight <br/>
+ x = index of the neuron which the weight is connected to
+ [-x] = x layers away from the output
<br/>

I use a proccess called backpropagation to adjust the weights.

After an images has gone trough the neural net, the output is very likely to be completely false, since the weights are initialized randomly.
To set give this falseness a value we use the cost function.

![Cost function](https://latex.codecogs.com/svg.image?\color{white}&space;C(...)&space;=&space;\sum_{n=0}^{9}(output[n]&space;-&space;desiredOutput[n])^{2})

Now we need to find out how sensitive the cost is to small changes to each weight, in order to adjust it.
In other words: find the partial-derivative of the cost function in respect to the weight.

This can easily be done by applying the chain rule.<br/>

### Weights between the hidden- and output-layer
![](https://latex.codecogs.com/svg.image?\color{white}\frac{\partial&space;C}{\partial&space;w}&space;=&space;\frac{\partial&space;C}{\partial&space;n}\frac{\partial&space;n}{\partial&space;z}\frac{\partial&space;z}{\partial&space;w})
<br/>

Now we have 3 simpler equations to solve. <br/>

![](https://latex.codecogs.com/svg.image?\color{white}\frac{\partial&space;C}{\partial&space;n}&space;=&space;2(output[i]&space;-&space;desiredOutput[i]))
<br/><br/>
![](https://latex.codecogs.com/svg.image?\color{white}\frac{\partial&space;n}{\partial&space;z}&space;=&space;sigmoid'(z))
<br/><br/>
![](https://latex.codecogs.com/svg.image?\color{white}\frac{\partial&space;z}{\partial&space;w}&space;=&space;n[-1])
<br/><br/>

Putting everything together...

![](https://latex.codecogs.com/svg.image?\color{white}\frac{\partial&space;C}{\partial&space;w}&space;=&space;2(output[i]&space;-&space;desiredOutput[i])sigmoid'(z)n[-1])

The next step is to subtract a small fraction of the just calculated value from the corresponding weight.<br/><br/>
![](https://latex.codecogs.com/svg.image?\color{white}w&space;=&space;w&space;-&space;lr\frac{\partial&space;C}{\partial&space;w})










