## Neural Network
The neural network consists of 3 layers:
+ input-layer: 784 neurons  --> pixels of the 28\*28 image
+ hidden-layer: 24 neurons  
+ output-layer: 10 neurons  --> matching with the numbers (0 - 9)

### Backpropagation
I use a proccess called backpropagation to adjust the weights.

After an images has gone trough the neural net, the output is very likely to be completely false, since the weights are initialized randomly.
To set give this falseness a value we use the cost function.

![Cost function](https://latex.codecogs.com/svg.image?C(...)&space;=&space;\sum_{n=0}^{9}output[n]&space;-&space;desiredOutput[n]&space;)
