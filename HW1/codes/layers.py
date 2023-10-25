import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor

class Relu(Layer):
	def __init__(self, name):
		super(Relu, self).__init__(name)

	def forward(self, input):
		self._saved_for_backward(input)
		return np.maximum(0, input)

	def backward(self, grad_output):
		input = self._saved_tensor
		return grad_output * (input > 0)

class Sigmoid(Layer):
	def __init__(self, name):
		super(Sigmoid, self).__init__(name)

	def forward(self, input):
		output = 1 / (1 + np.exp(-input))
		self._saved_for_backward(output)
		return output

	def backward(self, grad_output):
		output = self._saved_tensor
		return grad_output * output * (1 - output)

class Selu(Layer):
    def __init__(self, name):
        super(Selu, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        lamb, alpha = 1.0507, 1.67326
        output = lamb * np.where(input > 0, input, alpha * (np.exp(input) - 1))
        saved = lamb * np.where(input > 0, 1, output + alpha)
        self._saved_for_backward(saved)
        return output
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        return grad_output * self._saved_tensor
        # TODO END

class Swish(Layer):
    def __init__(self, name):
        super(Swish, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        sigm = 1 / (1 + np.exp(-input))
        output = sigm * input
        self._saved_for_backward(sigm + output * (1 - sigm))
        return output
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        return grad_output * self._saved_tensor
        # TODO END

class Gelu(Layer):
    def __init__(self, name):
        super(Gelu, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        alpha = np.sqrt(2/np.pi)
        beta = alpha * 0.044715
        x1 = alpha * input
        x2 = beta * input ** 3
        temp = np.tanh(x1 + x2)
        self._saved_for_backward(0.5 * (1 + temp + (x1 + 3 * x2) * (1 - temp ** 2)))
        output = 0.5 * input * (1 + temp)
        return output
        # TODO END
    
    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        return grad_output * self._saved_tensor
        # TODO END

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        self._saved_for_backward(input)
        return np.dot(input, self.W) + self.b
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        self.grad_W = np.dot(self._saved_tensor.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.W.T)
        return grad_input
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
