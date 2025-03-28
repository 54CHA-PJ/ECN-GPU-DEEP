# %%
import os
import math
import numpy as np
from numba import cuda
from tqdm import tqdm
import time

# %%
def make_uint32(b):
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]

def read_labels(filename):
    with open(filename, 'rb') as f:
        _ = f.read(4)
        n = make_uint32(f.read(4))
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    return data

def read_images(filename):
    with open(filename, 'rb') as f:
        _ = f.read(4)
        n = make_uint32(f.read(4))
        rows = make_uint32(f.read(4))
        cols = make_uint32(f.read(4))
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
        data = data.reshape(n, rows * cols)
    return data

# %%
@cuda.jit(device=True)
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

@cuda.jit(device=True)
def dsigmoid(x):
    s = sigmoid(x)
    return s * (1.0 - s)

@cuda.jit
def tiled_forward_layer_kernel(W, a_in, bias, z_out, a_out):
    TILE_SIZE = 16
    sW = cuda.shared.array((16, 16), dtype=np.float64)
    sA = cuda.shared.array((16, 16), dtype=np.float64)
    row = cuda.blockIdx.y * TILE_SIZE + cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE_SIZE + cuda.threadIdx.x
    sum_val = 0.0
    K = W.shape[1]
    for m in range((K + TILE_SIZE - 1) // TILE_SIZE):
        if row < W.shape[0] and m * TILE_SIZE + cuda.threadIdx.x < K:
            sW[cuda.threadIdx.y, cuda.threadIdx.x] = W[row, m * TILE_SIZE + cuda.threadIdx.x]
        else:
            sW[cuda.threadIdx.y, cuda.threadIdx.x] = 0.0
        if m * TILE_SIZE + cuda.threadIdx.y < K and col < a_in.shape[1]:
            sA[cuda.threadIdx.y, cuda.threadIdx.x] = a_in[m * TILE_SIZE + cuda.threadIdx.y, col]
        else:
            sA[cuda.threadIdx.y, cuda.threadIdx.x] = 0.0
        cuda.syncthreads()
        for k in range(TILE_SIZE):
            sum_val += sW[cuda.threadIdx.y, k] * sA[k, cuda.threadIdx.x]
        cuda.syncthreads()
    if row < z_out.shape[0] and col < z_out.shape[1]:
        z_val = sum_val + bias[row, 0]
        z_out[row, col] = z_val
        a_out[row, col] = 1.0 / (1.0 + math.exp(-z_val))

@cuda.jit
def output_backward_kernel(a, Y, delta):
    i, j = cuda.grid(2)
    if i < a.shape[0] and j < a.shape[1]:
        delta[i, j] = (a[i, j] - Y[i, j]) * (a[i, j] * (1.0 - a[i, j]))

@cuda.jit
def hidden_backward_kernel(W_next, delta_next, z, delta):
    i, j = cuda.grid(2)
    if i < z.shape[0] and j < z.shape[1]:
        s = 0.0
        for k in range(W_next.shape[0]):
            s += W_next[k, i] * delta_next[k, j]
        delta[i, j] = s * dsigmoid(z[i, j])

@cuda.jit
def update_weights_kernel(W, delta, a_prev, learning_rate, minibatch):
    i, j = cuda.grid(2)
    if i < W.shape[0] and j < W.shape[1]:
        grad = 0.0
        for k in range(delta.shape[1]):
            grad += delta[i, k] * a_prev[j, k]
        W[i, j] -= learning_rate / minibatch * grad

@cuda.jit
def update_biases_kernel(bias, delta, learning_rate, minibatch):
    i = cuda.grid(1)
    if i < bias.shape[0]:
        s = 0.0
        for k in range(delta.shape[1]):
            s += delta[i, k]
        bias[i, 0] -= learning_rate / minibatch * s

# %%
def grid_2d(shape, block=(16, 16)):
    gx = (shape[0] + block[0] - 1) // block[0]
    gy = (shape[1] + block[1] - 1) // block[1]
    return (gx, gy)

def grid_1d(n, block=256):
    return ((n + block - 1) // block,)

# %%
class LayerGPU:
    def __init__(self, n_in, n_out, minibatch):
        W_host = np.random.normal(0, 1/np.sqrt(n_in), (n_out, n_in)).astype(np.float64)
        self.W = cuda.to_device(W_host)
        self.b = cuda.to_device(np.zeros((n_out, 1), dtype=np.float64))
        self.z = cuda.device_array((n_out, minibatch), dtype=np.float64)
        self.a = cuda.device_array((n_out, minibatch), dtype=np.float64)
        self.delta = cuda.device_array((n_out, minibatch), dtype=np.float64)

class ANN:
    def __init__(self, layer_sizes, alpha, minibatch):
        self.alpha = alpha
        self.minibatch = minibatch
        self.n_layers = len(layer_sizes)
        self.layers = []
        for i in range(self.n_layers - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1]
            self.layers.append(LayerGPU(n_in, n_out, minibatch))
    def forward(self, X):
        a_in = cuda.to_device(X)
        block = (16, 16)
        for layer in self.layers:
            grid = grid_2d(layer.z.shape, block)
            tiled_forward_layer_kernel[grid, block](layer.W, a_in, layer.b, layer.z, layer.a)
            a_in = layer.a
        out = a_in.copy_to_host()
        return out
    def backward(self, X, Y):
        block = (16, 16)
        output_layer = self.layers[-1]
        Y_dev = cuda.to_device(Y)
        grid = grid_2d(output_layer.a.shape, block)
        output_backward_kernel[grid, block](output_layer.a, Y_dev, output_layer.delta)
        for l in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[l]
            next_layer = self.layers[l+1]
            grid = grid_2d(current_layer.a.shape, block)
            hidden_backward_kernel[grid, block](next_layer.W, next_layer.delta, current_layer.z, current_layer.delta)
        a_prev = cuda.to_device(X)
        for layer in self.layers:
            grid_w = grid_2d(layer.W.shape, block)
            grid_b = grid_1d(layer.b.shape[0], block=256)
            stream_w = cuda.stream()
            update_weights_kernel[grid_w, block, stream_w](layer.W, layer.delta, a_prev, self.alpha, self.minibatch)
            stream_b = cuda.stream()
            update_biases_kernel[grid_b, 256, stream_b](layer.b, layer.delta, self.alpha, self.minibatch)
            stream_w.synchronize()
            stream_b.synchronize()
            a_prev = layer.a
    def predict(self, X):
        out = self.forward(X)
        return np.argmax(out, axis=0)

# %%
def one_hot(labels, n_classes=10):
    arr = np.zeros((n_classes, len(labels)), dtype=np.float64)
    arr[labels, np.arange(len(labels))] = 1.0
    return arr

def cross_entropy_cpu(predictions, targets):
    eps = 1e-12
    clipped = np.clip(predictions, eps, 1 - eps)
    return -np.sum(targets * np.log(clipped)) / predictions.shape[1]

def accuracy(model, X, labels):
    batch_size = model.minibatch
    num_samples = X.shape[1]
    correct = 0
    for i in range(0, num_samples, batch_size):
        x_batch = X[:, i:i+batch_size]
        preds = model.predict(x_batch)
        correct += np.sum(preds == labels[i:i+batch_size])
    return correct / num_samples

# %%
DATA_PATH = "DATA"
train_img = read_images(os.path.join(DATA_PATH, "train-images.idx3-ubyte"))
train_label = read_labels(os.path.join(DATA_PATH, "train-labels.idx1-ubyte"))
test_img = read_images(os.path.join(DATA_PATH, "t10k-images.idx3-ubyte"))
test_label = read_labels(os.path.join(DATA_PATH, "t10k-labels.idx1-ubyte"))
train_img = train_img.astype(np.float64) / 255.0
test_img = test_img.astype(np.float64) / 255.0
train_img_pinned = cuda.pinned_array(train_img.shape, dtype=np.float64)
train_img_pinned[:] = train_img[:]
train_img = train_img_pinned
test_img_pinned = cuda.pinned_array(test_img.shape, dtype=np.float64)
test_img_pinned[:] = test_img[:]
test_img = test_img_pinned
alpha = 0.05
batch_size = 16
layer_sizes = [784, 30, 10]
epochs = 5
net = ANN(layer_sizes, alpha, batch_size)
Xtest = test_img.T
Ytest = test_label
init_acc = accuracy(net, Xtest, Ytest)
print(f"Initial Accuracy: {init_acc*100:.2f}%")
N = train_img.shape[0]
for ep in range(epochs):
    idx = np.arange(N)
    np.random.shuffle(idx)
    ce_total = 0.0
    n_batches = 0
    batch_iter = range(0, N - batch_size + 1, batch_size)
    acc_val = accuracy(net, Xtest, Ytest)
    desc = f'Epoch {ep+1} - Acc: {acc_val*100:.2f}%'
    for i in tqdm(batch_iter, desc=desc):
        bidx = idx[i:i+batch_size]
        x_b = train_img[bidx].T
        y_b = one_hot(train_label[bidx], 10)
        out = net.forward(x_b)
        ce = cross_entropy_cpu(out, y_b)
        ce_total += ce
        net.backward(x_b, y_b)
        n_batches += 1
    ce_mean = ce_total / n_batches
    acc_val = accuracy(net, Xtest, Ytest)
    print(f"Epoch {ep+1}: Accuracy = {acc_val*100:.2f}%, CE = {ce_mean:.4f}")
final_ce = ce_total / n_batches
final_acc = accuracy(net, Xtest, Ytest)
print("Final Model: Accuracy = {:.2f}%, Cross-Entropy Error = {:.4f}".format(final_acc*100, final_ce))
