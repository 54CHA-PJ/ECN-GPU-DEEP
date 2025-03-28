# %%
import os
import math
import numpy as np
from numba import cuda
from tqdm import tqdm

# %% [markdown]
# # Minimal MNIST Readers (CPU)

# %%
def make_uint32(b):
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]

def read_labels(filename):
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic
        n = make_uint32(f.read(4))
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    return data

def read_images(filename):
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic
        n = make_uint32(f.read(4))
        rows = make_uint32(f.read(4))
        cols = make_uint32(f.read(4))
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
        data = data.reshape(n, rows * cols)
    return data

# %% [markdown]
# # GPU Kernels for Matrix Operations
# 
# We keep all major ops on the GPU:
# - Dot product (matrix multiplication)
# - Hadamard (elementwise product)
# - Subtract (elementwise)
# - Sigmoid (in-place)
# - dSigmoid
# - Transpose
# - Sum of columns
# - Scalar multiplication
# - Add bias (broadcast over columns)

# %%
@cuda.jit
def dot_product_kernel(A, B, C):
    """
    C = A x B
    A.shape = (m, k)
    B.shape = (k, n)
    C.shape = (m, n)
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for kk in range(A.shape[1]):
            tmp += A[i, kk] * B[kk, j]
        C[i, j] = tmp

@cuda.jit
def hadamard_kernel(A, B, C):
    """
    C = A * B (elementwise)
    """
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < A.shape[1]:
        C[i, j] = A[i, j] * B[i, j]

@cuda.jit
def subtract_kernel(A, B, C):
    """
    C = A - B (elementwise)
    """
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < A.shape[1]:
        C[i, j] = A[i, j] - B[i, j]

@cuda.jit
def sigmoid_kernel(A):
    """
    In-place Sigmoid: A[i,j] = 1/(1+exp(-A[i,j]))
    """
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < A.shape[1]:
        A[i, j] = 1.0 / (1.0 + math.exp(-A[i, j]))

@cuda.jit
def dsigmoid_kernel(Z, Out):
    """
    Out[i,j] = derivative of sigmoid at Z[i,j].
    Recomputes s = 1/(1+exp(-Z[i,j])) then s*(1-s).
    """
    i, j = cuda.grid(2)
    if i < Z.shape[0] and j < Z.shape[1]:
        s = 1.0 / (1.0 + math.exp(-Z[i, j]))
        Out[i, j] = s * (1.0 - s)

@cuda.jit
def transpose_kernel(A, T):
    """
    T = A^T
    A.shape = (r, c)
    T.shape = (c, r)
    """
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < A.shape[1]:
        T[j, i] = A[i, j]

@cuda.jit
def sum_columns_kernel(A, column_sum):
    """
    Sums each row's elements across columns.
    A.shape = (m, n), column_sum.shape = (m, 1)
    column_sum[i,0] = sum over j of A[i,j]
    """
    i = cuda.grid(1)
    if i < A.shape[0]:
        s = 0.0
        for j in range(A.shape[1]):
            s += A[i, j]
        column_sum[i, 0] = s

@cuda.jit
def scalar_mul_kernel(A, scalar, Out):
    """
    Out = A * scalar (elementwise)
    """
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < A.shape[1]:
        Out[i, j] = A[i, j] * scalar

@cuda.jit
def add_bias_kernel(A, bias, Out):
    """
    Broadcast bias (shape (m,1)) across columns of A (shape (m,n)).
    Out[i,j] = A[i,j] + bias[i,0]
    """
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < A.shape[1]:
        Out[i, j] = A[i, j] + bias[i, 0]

# %% [markdown]
# # GPU Helper Functions

# %%
def grid_2d(shape, block=(16,16)):
    """
    Return a 2D grid for shape (rows, cols).
    """
    gx = (shape[0] + block[0] - 1) // block[0]
    gy = (shape[1] + block[1] - 1) // block[1]
    return (gx, gy)

def dot_product_gpu(A, B):
    """
    GPU: return A@B
    """
    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    C = np.empty((A.shape[0], B.shape[1]), dtype=np.float64)
    C_gpu = cuda.device_array_like(C)
    block = (16,16)
    grid = grid_2d(C.shape, block)
    dot_product_kernel[grid, block](A_gpu, B_gpu, C_gpu)
    C_gpu.copy_to_host(C)
    return C

def hadamard_gpu(A, B):
    """
    GPU: C = A * B elementwise
    """
    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    out = np.empty_like(A)
    out_gpu = cuda.device_array_like(out)
    block = (16,16)
    grid = grid_2d(A.shape, block)
    hadamard_kernel[grid, block](A_gpu, B_gpu, out_gpu)
    out_gpu.copy_to_host(out)
    return out

def subtract_gpu(A, B):
    """
    GPU: C = A - B elementwise
    """
    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    out = np.empty_like(A)
    out_gpu = cuda.device_array_like(out)
    block = (16,16)
    grid = grid_2d(A.shape, block)
    subtract_kernel[grid, block](A_gpu, B_gpu, out_gpu)
    out_gpu.copy_to_host(out)
    return out

def sigmoid_gpu(A):
    """
    GPU: in-place sigmoid of A
    Returns new array
    """
    A_gpu = cuda.to_device(A)
    block = (16,16)
    grid = grid_2d(A.shape, block)
    sigmoid_kernel[grid, block](A_gpu)
    return A_gpu.copy_to_host()

def dsigmoid_gpu(Z):
    """
    GPU: derivative of sigmoid w.r.t. Z
    Returns array of the same shape
    """
    Z_gpu = cuda.to_device(Z)
    out = np.empty_like(Z)
    out_gpu = cuda.device_array_like(out)
    block = (16,16)
    grid = grid_2d(Z.shape, block)
    dsigmoid_kernel[grid, block](Z_gpu, out_gpu)
    out_gpu.copy_to_host(out)
    return out

def transpose_gpu(A):
    """
    GPU: T = A^T
    """
    T = np.empty((A.shape[1], A.shape[0]), dtype=A.dtype)
    A_gpu = cuda.to_device(A)
    T_gpu = cuda.device_array_like(T)
    block = (16,16)
    grid = grid_2d(A.shape, block)
    transpose_kernel[grid, block](A_gpu, T_gpu)
    T_gpu.copy_to_host(T)
    return T

def sum_columns_gpu(A):
    """
    GPU: return sum of each row in A, result shape (A.shape[0], 1)
    """
    A_gpu = cuda.to_device(A)
    out = np.zeros((A.shape[0], 1), dtype=A.dtype)
    out_gpu = cuda.device_array_like(out)
    threads = 256
    blocks = (A.shape[0] + threads - 1)//threads
    sum_columns_kernel[blocks, threads](A_gpu, out_gpu)
    out_gpu.copy_to_host(out)
    return out

def scalar_multiply_gpu(A, scalar):
    """
    GPU: out = A * scalar
    """
    A_gpu = cuda.to_device(A)
    out = np.empty_like(A)
    out_gpu = cuda.device_array_like(out)
    block = (16,16)
    grid = grid_2d(A.shape, block)
    scalar_mul_kernel[grid, block](A_gpu, scalar, out_gpu)
    out_gpu.copy_to_host(out)
    return out

def add_bias_gpu(A, bias):
    """
    GPU: Out = A + bias (broadcast across columns)
    A.shape = (m, n), bias.shape = (m,1)
    """
    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(bias)
    out = np.empty_like(A)
    out_gpu = cuda.device_array_like(out)
    block = (16,16)
    grid = grid_2d(A.shape, block)
    add_bias_kernel[grid, block](A_gpu, B_gpu, out_gpu)
    out_gpu.copy_to_host(out)
    return out

# %% [markdown]
# # Small Neural Network (Everything GPU for main ops)

# %%
class Layer:
    def __init__(self, n_in, n_out, minibatch_size):
        """
        weights shape = (n_out, n_in)
        biases shape = (n_out, 1)
        a, z, delta shape = (n_out, minibatch_size)
        """
        self.weights = np.random.normal(0, 1/np.sqrt(n_in), (n_out, n_in))
        self.biases  = np.zeros((n_out, 1), dtype=np.float64)

        self.z     = np.zeros((n_out, minibatch_size), dtype=np.float64)
        self.a     = np.zeros((n_out, minibatch_size), dtype=np.float64)
        self.delta = np.zeros((n_out, minibatch_size), dtype=np.float64)

class ANN:
    def __init__(self, layer_sizes, alpha, minibatch):
        """
        layer_sizes: e.g. [784, 30, 10]
        alpha: learning rate
        minibatch: batch size
        """
        self.alpha = alpha
        self.minibatch = minibatch
        self.layers = []
        for i in range(len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], minibatch))

    def forward(self, X):
        """
        X shape = (n_in, batch_size)
        """
        # input is a(0)
        self.layers[0].a = X.copy()

        for i in range(len(self.layers)-1):
            l_in  = self.layers[i]
            l_out = self.layers[i+1]
            # z = W * a_in + bias
            ztmp = dot_product_gpu(l_out.weights, l_in.a)
            ztmp = add_bias_gpu(ztmp, l_out.biases)  # GPU broadcast add
            l_out.z = ztmp
            l_out.a = sigmoid_gpu(ztmp)  # GPU in-place sigmoid

        return self.layers[-1].a

    def backward(self, X, Y):
        """
        Y shape = (n_out, batch_size)
        """
        L = len(self.layers) - 1
        out_layer = self.layers[L]

        # delta_L = (a_L - Y) * dsigmoid(Z_L)
        diff = subtract_gpu(out_layer.a, Y)
        dsgm = dsigmoid_gpu(out_layer.z)
        out_layer.delta = hadamard_gpu(diff, dsgm)

        # propagate deltas backward
        for i in reversed(range(1, L)):
            curr = self.layers[i]     
            nxt  = self.layers[i+1]   
            wT = transpose_gpu(nxt.weights)
            dprop = dot_product_gpu(wT, nxt.delta)
            dsgm_i = dsigmoid_gpu(curr.z)
            curr.delta = hadamard_gpu(dprop, dsgm_i)

        # weight/bias updates
        for i in range(1, len(self.layers)):
            prev = self.layers[i-1]
            curr = self.layers[i]

            # dW = (delta_i @ a_{i-1}^T) * (alpha / batch_size)
            aT = transpose_gpu(prev.a)
            dW = dot_product_gpu(curr.delta, aT)
            scale = self.alpha / self.minibatch
            dW_scaled = scalar_multiply_gpu(dW, scale)
            curr.weights = subtract_gpu(curr.weights, dW_scaled)

            # dB = sum of delta_i across columns, then scale
            sumd = sum_columns_gpu(curr.delta)  # shape (n_out, 1)
            sumd_scaled = scalar_multiply_gpu(sumd, scale)
            curr.biases = subtract_gpu(curr.biases, sumd_scaled)

    def predict(self, X):
        # forward pass
        out = self.forward(X)
        # for classification, do CPU argmax
        return np.argmax(out, axis=0)

# %% [markdown]
# # Training / Testing with MNIST

# %%
def one_hot(labels, n_classes=10):
    arr = np.zeros((n_classes, len(labels)), dtype=np.float64)
    arr[labels, np.arange(len(labels))] = 1.0
    return arr

def accuracy(model, X, Y):
    """
    Evaluate classification accuracy
    X shape: (n_in, num_samples)
    Y shape: (num_samples,) with integer labels
    """
    batch_size = model.minibatch
    num_samples = X.shape[1]
    nbatches = (num_samples // batch_size)*batch_size
    correct = 0
    for i in range(0, nbatches, batch_size):
        xb = X[:, i:i+batch_size]
        yb = Y[i:i+batch_size]
        preds = model.predict(xb)
        correct += np.sum(preds == yb)
    return correct / nbatches

def cross_entropy_cpu(predictions, targets):
    """
    CPU cross-entropy
    predictions, targets shape: (n_out, batch_size)
    """
    eps = 1e-12
    clipped = np.clip(predictions, eps, 1-eps)
    return -np.sum(targets * np.log(clipped)) / predictions.shape[1]

# %%
if __name__ == "__main__":
    # Basic config
    DATA_PATH = "DATA"
    train_img = read_images(os.path.join(DATA_PATH, "train-images.idx3-ubyte"))
    train_label = read_labels(os.path.join(DATA_PATH, "train-labels.idx1-ubyte"))
    test_img  = read_images(os.path.join(DATA_PATH, "t10k-images.idx3-ubyte"))
    test_label = read_labels(os.path.join(DATA_PATH, "t10k-labels.idx1-ubyte"))

    # Normalize to [0,1]
    train_img = train_img.astype(np.float64) / 255.0
    test_img  = test_img.astype(np.float64)  / 255.0

    # Hyperparameters
    alpha = 0.05
    batch_size = 16
    layer_sizes = [784, 30, 10]
    epochs = 5

    # Build net
    net = ANN(layer_sizes, alpha, batch_size)

    # Quick check on 1000 test samples
    Xtest = test_img[:1000].T  # shape (784, 1000)
    Ytest = test_label[:1000]

    init_acc = accuracy(net, Xtest, Ytest)
    print(f"Initial Accuracy: {init_acc*100:.2f}%")

    N = train_img.shape[0]
    for ep in range(epochs):
        idx = np.arange(N)
        np.random.shuffle(idx)
        ce_total = 0.0
        n_batches = 0

        for i in tqdm(range(0, N - batch_size + 1, batch_size)):
            bidx = idx[i:i+batch_size]
            x_b  = train_img[bidx].T  # shape (784, batch_size)
            y_b  = one_hot(train_label[bidx], 10)

            # Forward
            out = net.forward(x_b)
            # Cross-entropy on CPU
            ce = cross_entropy_cpu(out, y_b)
            ce_total += ce

            # Backward
            net.backward(x_b, y_b)
            n_batches += 1

        acc_val = accuracy(net, Xtest, Ytest)
        print(f"Epoch {ep+1}, Accuracy: {acc_val*100:.2f}%, CE: {ce_total/n_batches:.4f}")

    final_acc = accuracy(net, Xtest, Ytest)
    print(f"Final Test Accuracy on 1000 samples: {final_acc*100:.2f}%")
