## Neural Network
The neural network consists of 3 layers:
+ input-layer: 784 neurons  --> pixels of the 28\*28 image
+ hidden-layer: 24 neurons
+ output-layer: 10 neurons  --> represents the matching with the numbers from 0 to 9
<br/> 
As activation function, sigmoid is used.

## Backpropagation
I use a process called backpropagation to adjust the weights.

After the first images has gone trough the neural net, the output is very likely to be completely false, since the weights were initialized randomly.
To give this falseness a value we use the cost function.

$$C(...)=\sum_{n=0}^{9}(output[n]-desiredOutput[n])^{2}$$
### Adjusting the weights between the hidden- and the output-layer

For the sake of simplicity, let's first look at **one** weight and call it `wx`. It creates a connection between `nh` (neuron somewhere in the hidden layer) and `no` (neuron somewhere in the output layer).<br/>

Now we need to find out how sensitive the cost is to small changes in `wx`, in order to adjust it. <br/>
In other words: find the partial-derivative of the cost function with respect to `wx`.<br/>

$$\frac{\partial C}{\partial wx}=?$$
 
This can easily be done by applying the chain rule (because we could see every step of forwardpropagation as a function). Note: `zo` = `no` before the sigmoid function was applied.<br/>

$$\frac{\partial C}{\partial wx}=\frac{\partial C}{\partial no}\frac{\partial no}{\partial zo}\frac{\partial zo}{\partial wx}$$

$$\frac{\partial C}{\partial no} = 2(no - desiredOutput)$$ <br/>
$$\frac{\partial no}{\partial zo} = sigmoid'(zo)$$ <br/>
$$\frac{\partial zo}{\partial wx} = nh$$ <br/>

Putting everything together...

$$\frac{\partial C}{\partial wx} = 2(no - desiredOutput)sigmoid'(zo)nh$$

The next step is to multiply the just calculated value with a learning rate `lr = 0.01` and subtract it from `wx`. <br/>
We use this learning rate to prevent the net from learning too fast, because it leads to bad results.


$$wx = wx - lr * 2(no - desiredOutput)sigmoid'(zo)nh$$

This proccess is then repeated for every other weight, that is connecting the hidden- with the output-layer.

### Adjusting the weights between the input- and the hidden-layer
Let's once again look at just one weight and call it `wy`. It connects `ni` (neuron somewhere in the input layer) and `nh` (neuron somewhere in the hidden layer).
This procedure is very similar to the one above. The only difference is, that the partial derivative of the cost with respect to `nh` is not as easy to calculate. Since `nh` is connected to every neuron of the output-layer, we have to sum up the impact it has trough each weight-connection to the output layer.

Further notations: `o` = output-layer, `z` = output-layer before sigmoid was applied, `zh` = `nh` before sigmoid was applied <br/><br/>
$$\frac{\partial C}{\partial nh} =  \sum_{i=0}^{9} \frac{\partial C}{\partial o_{i}} \frac{\partial o_{i}}{\partial z_{i}} \frac{\partial z_{i}}{\partial nh} $$
$$\frac{\partial C}{\partial nh} = \sum_{i=0}^{9} 2(o_{i} - desiredOutput_{i})sigmoid'(z_{i})nh$$

Now we can just apply the chain rule like previously. <br/>


$$\frac{\partial C}{\partial wy} = \frac{\partial C}{\partial nh} \frac{\partial nh}{\partial zh} \frac{\partial zh}{\partial wy}$$
$$\frac{\partial C}{\partial wy} = (\sum_{i=0}^{9} 2(o_{i} - desiredOutput_{i})sigmoid'(z_{i})nh) * sigmoid'(zh) * ni$$

Subtract a fraction of this value from `wy`... 

$$w = w - lr(\sum_{i=0}^{9} 2(o_{i} - desiredOutput_{i})sigmoid'(z_{i})nh)sigmoid'(zh)ni)$$

Repeat this procedure for every weight connecting the input- with the hidden-layer and we're done!
With my configuration I do this 3 times for 50000 images each.

## Results
![Results](https://drive.google.com/uc?export=view&id=1CB1EgtA3UfGmTb0EVjMIlMqYsjTvuqwR)




