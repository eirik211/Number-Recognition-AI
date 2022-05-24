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

Further notations: `o` = output-layer, `z` = output-layer before sigmoid was applied, `zh` = `nh` before sigmoid was applied <br/><br/>
![](https://latex.codecogs.com/svg.image?\color{white}&space;\frac{\partial&space;C}{\partial&space;nh}&space;=&space;&space;\sum_{i=0}^{9}&space;\frac{\partial&space;C}{\partial&space;o_{i}}&space;\frac{\partial&space;o_{i}}{\partial&space;z_{i}}&space;\frac{\partial&space;z_{i}}{\partial&space;nh}&space;)

Now we can just apply the chain rule like previously. <br/>


![](https://latex.codecogs.com/svg.image?\color{white}&space;\frac{\partial&space;C}{\partial&space;wy}&space;=&space;\frac{\partial&space;C}{\partial&space;nh}&space;\frac{\partial&space;nh}{\partial&space;zh}&space;\frac{\partial&space;zh}{\partial&space;wy})
<br/><br/>
![](https://latex.codecogs.com/svg.image?\color{white}&space;\frac{\partial&space;C}{\partial&space;wy}&space;=&space;(\sum_{i=0}^{9}\frac{\partial&space;C}{\partial&space;o_{i}}\frac{\partial&space;o_{i}}{\partial&space;z_{i}}\frac{\partial&space;z_{i}}{\partial&space;nh})sigmoid'(zh)ni)
<br/><br/>

Subtract a fraction of this value from `wy`... <br/><br/>

![](https://latex.codecogs.com/svg.image?\color{white}&space;w&space;=&space;w&space;-&space;lr(\sum_{i=0}^{9}\frac{\partial&space;C}{\partial&space;o_{i}}\frac{\partial&space;o_{i}}{\partial&space;z_{i}}\frac{\partial&space;z_{i}}{\partial&space;nh})sigmoid'(zh)ni)
<br/><br/>

Repeat this procedure for every weight connecting the input- with the hidden-layer and we're done!
With my configuration I do this 3 times for 50000 images each.

### Results
![]([https://drive.google.com/file/d/1CB1EgtA3UfGmTb0EVjMIlMqYsjTvuqwR/view?usp=sharing](https://doc-04-30-docs.googleusercontent.com/docs/securesc/q0l9qqfhtk6aik7e49rrf1kpmif7k3q7/l53lqu6skqjhb4i2umtksg9p512aa9ha/1653405000000/13041865046088965694/13041865046088965694/1CB1EgtA3UfGmTb0EVjMIlMqYsjTvuqwR?e=view&ax=ACxEAsYHxGi36SwJ-jsswgJnkybc4bwx8q87-csIe3C9buG58CLFr1sfhm8GGCVVGTICJ3lnyNSY6u5GF6uLMjnS8XGqSfztz2fTQOQe_3XioVAAsK263lJFUSjr9LPSC9ArohpvmfvhK8cjlkVQqoFx64xkmPgn0--gmM0nIUtLSEAvhr-kPbYm661cBoeIX-QmrIP6xFWqGXQvVxBskqcv6uAIk36uMAFRMjOau82b-3RgJ0UaikMrfhOBgDZzE-p0r7Dhf3ghOi1hQEc_QEq_hp-8JFCIhhqam24T8cYBbxmCFZaKkdXReGcxt4cVkXwr2W7MpsI6XeiGk_BmFd4vn-6oOnH4r4tb2TZsTz1FjHZSA_bU92EiiwsBYMq59LaaRyak7NY8AMCImP2LFs-gjxzuR1yiM_hUHPH3y_3qFk0ohqJNvrzQ6DOqi2QAi__GAOx3Pe85IRQ40FtIzxWwRO3pMyuzMPVUcSYUzzOhkyOMUwYpM3ohUOtpIp2W5ashdlZGJJf6xVsRiEHcM6NWa_w2OquVqTaLjzALrpP6zS_clkgxtoj1M-f396oIZ-XSpFpkpNytKa5k5OO_5GPnac6wGyaiSYdB3uC0CKRvJBcnl2Sn4ShGgzMSYC2gA4Krl6OW-wmoed2PYLRlkorwmItisQkYLOpc7v-k7Lo18Jzy_JiNToMwesVKywtGxQF6GgJvVh0K8RN9hYPu8OnNusD1c_o4LfxSuqwicXTqwfGFkLjBmoMXFOFk-HZetmmzp8z6txxZE5ryUHeZZ6RHra-Xzs0cCLizKKdLER946zc2HzRNJyupu8fr6Q&authuser=0))




