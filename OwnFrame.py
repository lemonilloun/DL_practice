import numpy as np

class Tensor (object):

    def __init__(self, data, autograd=False, creators=None, creation_op=None, id=None):
        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None
        self.autograd = autograd
        self.children = {}
        if id is None:
            self.id = np.random.randint(0, 100000)
        else:
            self.id = id

        # скорректировать число потомков данного тензора
        if creators is not None:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1


    # проверить, получил ли тензор градиенты для всех потомков
    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True


    #вычисление обратного распространения
    def backward(self, grad=None, grad_origin=None):
        if self.autograd:

            # первый градиентный вызов из единиц
            if grad is None:
                grad = Tensor(np.ones_like(self.data))

            if grad_origin is not None:
                # проверка возможности обратного распространения
                # или ожидания градиента,
                # в последнем случае нужно уменьшить счетчик
                if self.children[grad_origin.id] == 0:
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1

            # накопление градиентов от нескольких потомков
            if self.grad  is None:
                self.grad = grad
            else:
                self.grad += grad

            assert grad.autograd == False

            if (self.creators is not None) and (self.all_children_grads_accounted_for() or grad_origin is None):

                if self.creation_op == 'add':
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self) # начало обратного распространения

                if self.creation_op == "neg":
                    self.creators[0].backward(self.grad.__neg__())

                if self.creation_op == "sub":
                    new = Tensor(self.grad.data)
                    self.creators[0].backward(new, self)
                    new = Tensor(self.grad.__neg__().data)
                    self.creators[1].backward(new, self)

                if self.creation_op == "mul":
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new, self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)

                if self.creation_op == "mm":
                    act = self.creators[0] # обычно слой активации
                    weights = self.creators[1] # обычно весовая матрица
                    new = self.grad.mm(weights.transpose())
                    act.backward(new)
                    new = self.grad.transpose().mm(act).transpose()
                    weights.backward(new)

                if self.creation_op == "transpose":
                    self.creators[0].backward(self.grad.transpose())

                if "sum" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    ds = self.creators[0].data.shape[dim]
                    self.creators[0].backward(self.grad.expand(dim, ds))

                if "expand" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))

                if self.creation_op == "sigmoid":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self * (ones - self)))

                if self.creation_op == "tanh":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (ones - (self * self)))

                if self.creation_op == "index_select":
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices_ = self.index_select_indices.data.flatten() # преобразование индексов в плоский вектор
                    grad_ = grad.data.reshape(len(indices_), -1) # свораичиваем гадиент в простой список строк

                    #  на данном моменте индексы indices_ и grad_ будут иметь соответсвующий порядок

                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad))

                if self.creation_op == "cross_entropy":
                    dx = self.softmax_output - self.target_dist
                    self.creators[0].backward(Tensor(dx))


    # реализация метода сложения
    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, autograd=True,
                          creators=[self, other],
                          creation_op='add')
        return Tensor(self.data + other.data, creators=[self,other], creation_op='add')

    # реализация метода отрицания (negative)
    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        return Tensor(self.data * -1)

    # реализация метода вычитания
    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="sub")
        return Tensor(self.data - other.data)

    # реализация метода умножения
    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="mul")
        return Tensor(self.data * other.data)

    # реализация метода суммирования всех эллементов массива
    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_" + str(dim),)
        return Tensor(self.data.sum(dim))

    # реализация метода расширения
    def expand(self, dim, copies):

        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        if self.autograd:
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_" + str(dim))
        return Tensor(new_data)

    #  реализация метода транспонирования
    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")
        return Tensor(self.data.transpose())

    # реализация матричного умножения
    def mm(self, x):
        if self.autograd:
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self, x],
                          creation_op="mm")
        return Tensor(self.data.dot(x.data))

    # функции активации нейронов

    def sigmoid(self):
        if self.autograd:
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op='sigmoid')
        return Tensor(1 / (1 + np.exp(-self.data)))


    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op="tanh")
        return Tensor(np.tanh(self.data))

    def index_select(self, indices):

        if self.autograd:
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         creators=[self],
                         creation_op="index_select")
            new.index_select_indices = indices
            return new

        return Tensor(self.data[indices.data])

    def cross_entropy(self, target_indices):

        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape) - 1,
                                       keepdims=True)

        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()

        if self.autograd:
            out = Tensor(loss,
                         autograd=True,
                         creators=[self],
                         creation_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out

        return Tensor(loss)

    def softmax(self):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape)-1,
                                       keepdims=True)
        return softmax_output

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())

class SGD (object):

    def __init__(self, parameters, alpha = 0.1):
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero=True):
        for p in self.parameters:
            p.data -= p.grad.data * self.alpha

            if zero:
                p.grad.data *= 0

class Layer (object):
    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters

class Linear(Layer):

    def __init__(self, n_inputs, n_outputs, bias=True):
        super().__init__()
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        self.weight = Tensor(W, autograd=True)
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input):
        return input.mm(self.weight) + self.bias.expand(0, len(input.data))

class Sequential(Layer):

    def __init__(self, layers=list()):
        super().__init__()
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params

class MSELoss (Layer):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred - target) * (pred - target)).sum(0)

class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sigmoid()

class Embedding(Layer):

    def __init__(self, vocab_size, dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim

        weight = (np.random.rand(vocab_size, dim) - 0.5) / dim
        self.weight = Tensor(weight, autograd=True)

        self.parameters.append(self.weight)

    def forward(self, input):
        return self.weight.index_select(input)

class CrossEntropyLoss(object):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return input.cross_entropy(target)

class RNNCell(Layer):
    def __init__(self, n_input, n_hidden, n_output, activation='sigmoid'):
        super().__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        if activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'tanh':
            self.activation = Tanh()
        else:
            raise Exception("Non-linearity not found")

        self.w_ih = Linear(n_input, n_hidden)
        self.w_hh = Linear(n_hidden, n_hidden)
        self.w_ho = Linear(n_hidden, n_output)


        self.parameters += self.w_ih.get_parameters()
        self.parameters += self.w_hh.get_parameters()
        self.parameters += self.w_ho.get_parameters()

    def forward(self, input, hidden):
        from_prev_hidden = self.w_hh.forward(hidden)
        combined = self.w_ih.forward(input) + from_prev_hidden
        new_hidden = self.activation.forward(combined)
        output = self.w_ho.forward(new_hidden)

        return output, new_hidden

    def init_hidden(self, batch_size):
        return Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)

# слой долгой краткосрочной памяти
class LSTMCell(Layer):

    def __init__(self, n_inputs, n_hidden, n_output):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

        # собственные весовые матрицы для вентилей

        # входные весовые матрицы
        self.xf = Linear(n_inputs, n_hidden)
        self.xi = Linear(n_inputs, n_hidden)
        self.xo = Linear(n_inputs, n_hidden)
        self.xc = Linear(n_inputs, n_hidden)

        # скрытые весовые матрицы
        self.hf = Linear(n_hidden, n_hidden, bias=False)
        self.hi = Linear(n_hidden, n_hidden, bias=False)
        self.ho = Linear(n_hidden, n_hidden, bias=False)
        self.hc = Linear(n_hidden, n_hidden, bias=False)

        # выходная матрица весов
        self.w_ho = Linear(n_hidden, n_output, bias=False)

        self.parameters += self.xf.get_parameters()
        self.parameters += self.xi.get_parameters()
        self.parameters += self.xo.get_parameters()
        self.parameters += self.xc.get_parameters()
        self.parameters += self.hf.get_parameters()
        self.parameters += self.hi.get_parameters()
        self.parameters += self.ho.get_parameters()
        self.parameters += self.hc.get_parameters()

        self.parameters += self.w_ho.get_parameters()

    def forward(self, input, hidden):

        prev_hidden, prev_cell = hidden[0], hidden[1]

        # вентили
        forget = (self.xf.forward(input) + self.hf.forward(prev_hidden)).sigmoid()
        inp = (self.xi.forward(input) + self.hi.forward(prev_hidden)).sigmoid()
        outp = (self.xo.forward(input) + self.ho.forward(prev_hidden)).sigmoid()
        update = (self.xc.forward(input) + self.hc.forward(prev_hidden)).sigmoid()

        # сумма предыдущего значения prev_cell и приращения uddate
        # контролируемые взвешенными весами forget и inp
        cell = (forget * prev_cell) + (inp * update)

        # замаскированная версия ячейки cell
        hidden = outp * cell.tanh()

        output = self.w_ho.forward(hidden)

        return output, (hidden, cell)

