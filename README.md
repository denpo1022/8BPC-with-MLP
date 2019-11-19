# Learning 8-bit parity checking problem with MLP

## Description and Explanation of Code

Basically the code can be seperated into roughly three parts:

1. **Training Data Generator Function** `InputGenerator(inputs)`: This function can generate the taining data and the answer we need.
2. **MLP Model Designing** `MLP(object)`: The core part of the hoemwork this time. Including parameters initialization, model sequences, forward and backward function, weight and bias update, data training and error calculating.
3. **Enterpoint Main Function** `main()`: The main function where we assign the specific value of variant parameter into MLP model and try out the best number to let the error minimized.

---

### Training Data Generator Function

``` python=135
def InputGenerator(inputs):
    # referrence from https://en.wikipedia.org/wiki/Hamming_weight
    # and http://p-nand-q.com/python/algorithms/math/bit-parity.html

    def BitParityCal(i):
        i = i - ((i >> 1) & 0x55555555)
        i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
        i = (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24
        return int(i % 2)

    data = []
    answer = []
    for i in range(inputs):
        data.append(np.asarray(list(f'{i:08b}'), dtype=np.int32))
        if BitParityCal(i) == 1:
            answer.append([1, 0])
        else:
            answer.append([0, 1])

    data = np.array(data)
    answer = np.array(answer)

    output = []
    for i in range(data.shape[0]):
        tupledata = list((data[i, :].tolist(), answer[i].tolist()))
        output.append(tupledata)

    return output
```

#### Important points in this section

1. Using Hamming weight method to calculate parity bit.

    > The Hamming weight of a string is the number of     symbols that are different from the zero-symbol of the   alphabet used. It is thus equivalent to the Hamming distance from the all-zero string of the same length. For the most typical case, a string of bits, this is the number of 1's in the string, or the digit sum of the binary representation of a given number and the ℓ₁ norm of a bit vector. In this binary case, it is also called the population count, popcount, sideways sum, or bit summation. [From https://en.wikipedia.org/wiki/Hamming_weight](https://en.wikipedia.org/wiki/Hamming_weight)

2. Combine the training data with answer bit into the style like this: `[[[x1, x2, ..., xn], [y1, y2, ..., yn]], [[x1~xn], [y1~yn]], ...]` where x represents traning data, y represents parity bit. Example: `output[0]` has value `[[0, 0, 0, 0, 0, 0, 0, 0], [0, 1]]`, `output[1]` has value `[[0, 0, 0, 0, 0, 0, 0, 1], [1, 0]]` where `[0, 1]` stands for even amounts of parity bit and `[1, 0]` stands for odd amounts of parity bit.

---

### MLP Model Designing

#### Important points in this section

1. Designing of the acitvation function:`Sigmoid(x)`,`DerivativeSigmoid(y)`,`Tanh(x)`,`DerivativeTanh(y)`.

``` python=6
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def DerivativeSigmoid(y):
    return y * (1.0 - y)


def Tanh(x):
    return np.tanh(x)


def DerivativeTanh(y):
    return 1 - y * y
```

![sigmoid](https://i.imgur.com/VcqGf5U.png)![de_sigmoid](https://i.imgur.com/v6x4asF.png)![tanh](https://i.imgur.com/C6qHFjb.png)![de_tanh](https://i.imgur.com/KM58G55.png)

2. Initializing variant of parameters: `iterations`, `learning_rate`, `momentum`, `input`, `hidden` and `output` arrays, also using `np.random.normal()` to initialize weights.

``` python=23
def __init__(self, input, hidden, output, iterations=200, learning_rate=0.4, momentum=0.6):

        # initialize parameters
        self.iterations = iterations
        self.learning_rate = learning_rate

        # momentum is used for escaping from local minimum situation
        self.momentum = momentum

        # initialize arrays
        self.input = input + 1  # add 1 for bias node
        self.hidden = hidden
        self.output = output

        # set up array of 1s for activations
        self.activation_input = np.ones(self.input)
        self.activation_hidden = np.ones(self.hidden)
        self.activation_output = np.ones(self.output)

        # create randomized weights
        input_range = 1.0 / self.input ** (1/2)
        self.weight_input = np.random.normal(
            loc=0, scale=input_range, size=(self.input, self.hidden))
        self.weight_output = np.random.uniform(
            size=(self.hidden, self.output)) / np.sqrt(self.hidden)

        # setup temporary array used for saving updated weights
        self.temporary_input = np.zeros((self.input, self.hidden))
        self.temporary_output = np.zeros((self.hidden, self.output))
```

3. Designing the forwarding function `feedForward(self, inputs)`: Send the inputs of activation through the order of `Tanh()` and `Sigmoid()`, finally return the output of activation.

``` python=53
def feedForward(self, inputs):

        # check the size of input is correct or not
        if len(inputs) != self.input-1:
            print(len(inputs), self.input-1)
            raise ValueError('Wrong number of inputs!')

        # input activations
        self.activation_input[0:self.input - 1] = inputs

        # hidden activations using Tanh()
        sum_up = np.dot(self.weight_input.T, self.activation_input)
        self.activation_hidden = Tanh(sum_up)

        # output activations using Sigmoid()
        sum_up = np.dot(self.weight_output.T, self.activation_hidden)
        self.activation_output = Sigmoid(sum_up)

        return self.activation_output
```

4. Designing the backwarding function `backPropagate(self, targets)`: First calculate the error value to both hidden and output layer. Then update the weights connecting the hidden and output layer. After that using mean squared error function to calculate the error value and return it.

``` python=73
def backPropagate(self, targets):

        # check the size of output is correct or not
        if len(targets) != self.output:
            print(len(targets), self.output)
            raise ValueError('Wrong number of targets!')

        # calculate error terms for output
        # delta tells which direction to change the weights
        output_deltas = DerivativeSigmoid(
            self.activation_output) * -(targets - self.activation_output)

        # calculate error terms for hidden
        # delta tells which direction to change the weights
        error = np.dot(self.weight_output, output_deltas)
        hidden_deltas = DerivativeTanh(self.activation_hidden) * error

        # update the weights connecting hidden to output, 'change' is partial derivative
        change = output_deltas * np.reshape(self.activation_hidden,
                                            (self.activation_hidden.shape[0], 1))
        self.weight_output -= self.learning_rate * \
            change + self.temporary_output * self.momentum
        self.temporary_output = change

        # update the weights connecting input to hidden, 'change' is partial derivative
        change = hidden_deltas * \
            np.reshape(self.activation_input,
                       (self.activation_input.shape[0], 1))
        self.weight_input -= self.learning_rate * \
            change + self.temporary_input * self.momentum
        self.temporary_input = change

        # calculate error (using mean squared error function)
        error = sum((1 / self.output) * (targets - self.activation_output)**2)

        return error
```

1. Training fuction and Showing the result figure: Send the training data into `trainAndShowError(self, train_data)`and shuffling the training data. Put the train data array to forwarding function, reciving the error value from back propagate function. Now we have error values from first to last iterations, and thus we can append the error array using these value. Finally show up the training error figure as result.

``` python=110
def trainAndShowError(self, train_data):
        num_example = np.shape(train_data)[0]

        Error = []
        for i in range(self.iterations):
            error = 0.0
            random.shuffle(train_data)
            for p in train_data:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error += self.backPropagate(targets)

            Error.append(error)

        # show the training error figure
        plt.figure()
        plt.plot(range(self.iterations), Error, 'r-', label="train error")
        plt.title('Perceptron training')
        plt.xlabel('Epochs')
        plt.ylabel('Training Error')
        plt.legend()
        plt.show()
```

---

### Enterpoint Main Function

What we do in `main()` is setting up the best parameter I hope (after many times try-and-error examination). With the help of the parameter `momentum` the relsult is not bad. We can see the training error almost reachs 0 after 150 epochs.

``` python=165
def main():

    # get input data from InputGenerator()
    X = InputGenerator(256)

    # input size = 8 (bits)
    input_size = 8

    # amounts of hidden layer
    hidden_layer = 20

    # output size = 2 ([1, 0] or [0, 1])
    output_size = 2

    # total of epoch
    iteration_times = 200

    # set learning rate
    learning_rate = 0.4

    # set momentum
    momentum = 0.6

    model = MLP(input_size, hidden_layer, output_size, iterations=iteration_times,
                learning_rate=learning_rate, momentum=momentum)

    model.trainAndShowError(X)
```

![result](https://i.imgur.com/polrAGa.png)
