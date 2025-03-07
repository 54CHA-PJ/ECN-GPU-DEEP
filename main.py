# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: ECN_GPU
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style
import math
import os


# %% [markdown]
# ---
# ## 1. Lecture des fichiers MNIST

# %%
def make_uint32(byte_array):
    """ Recompose un entier 32 bits à partir de 4 octets de poids fort à poids faible """
    return ((byte_array[0] << 24) 
          | (byte_array[1] << 16) 
          | (byte_array[2] <<  8) 
          | (byte_array[3] <<  0))

def read_labels(filename):
    """ Lit un fichier de labels MNIST """
    with open(filename, 'rb') as f:
        _ = f.read(4)  # Magic number (non utilisé)
        n_bytes = f.read(4)
        n = make_uint32(n_bytes)
        labels = np.frombuffer(f.read(n), dtype=np.uint8)
    return labels

def read_images(filename):
    """ Lit un fichier d'images MNIST """
    with open(filename, 'rb') as f:
        _ = f.read(4)  # Magic number (non utilisé)
        n_bytes = f.read(4)
        n = make_uint32(n_bytes)
        row_bytes = f.read(4)
        col_bytes = f.read(4)
        rows = make_uint32(row_bytes)
        cols = make_uint32(col_bytes)
        images_raw = f.read(n * rows * cols)
        images = np.frombuffer(images_raw, dtype=np.uint8)
        images = images.reshape(n, rows * cols)
    return images


# %% [markdown]
# ---
# ## 2. Fonctions utilitaires

# %%
def zero_to_n(n):
    """ Crée un tableau [0, 1, 2, ..., n-1] """
    return np.arange(n, dtype=np.uint32)

def shuffle(t, number_of_switch):
    """ Mélange un tableau t aléatoirement, en réalisant 'number_of_switch' échanges """
    size = len(t)
    for _ in range(number_of_switch):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        tmp = t[x]
        t[x] = t[y]
        t[y] = tmp
        
def init_sigma(nneurons_prev):
    return 1.0 / np.sqrt(nneurons_prev)  # REPLACE NP - MATH

def sigmoid(x):
    """ Fonction d'activation vectorisée """
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
    """ Dérivée de sigmoid vectorisée """
    s = sigmoid(x)
    return s * (1.0 - s)


# %% [markdown]
# ---
# ## 3. Fonctions Matricielles

# %%
def alloc_matrix(rows, columns):
    # En C, on fait un malloc puis un tableau 1D rows*columns.
    # En Python, on crée un np.ndarray de shape (rows, columns).
    return np.zeros((rows, columns), dtype=np.float64)


# %% [markdown]
# ### 3.1. Version Naïve
# (Utilise des boucles explicites pour imiter l'implémentation en C)

# %%
def matrix_dot_naive(m1, m2):
    # Produit entre deux matrices : triple boucle
    r1, c1 = m1.shape
    r2, c2 = m2.shape
    res = alloc_matrix(r1, c2)
    for i in range(r1):
        for j in range(c2):
            tmp = 0.0
            for k in range(c1):
                tmp += m1[i, k] * m2[k, j]
            res[i, j] = tmp
    return res

def matrix_sum_naive(m1, m2):
    # Addition entre deux matrices : double boucle
    r, c = m1.shape
    res = alloc_matrix(r, c)
    for i in range(r):
        for j in range(c):
            res[i, j] = m1[i, j] + m2[i, j]
    return res

def matrix_minus_naive(m1, m2):
    # Soustraction entre deux matrices (m1 - m2) : double boucle
    r, c = m1.shape
    res = alloc_matrix(r, c)
    for i in range(r):
        for j in range(c):
            res[i, j] = m1[i, j] - m2[i, j]
    return res

def hadamard_product_naive(m1, m2):
    # Produit d'Hadamard : double boucle
    r, c = m1.shape
    res = alloc_matrix(r, c)
    for i in range(r):
        for j in range(c):
            res[i, j] = m1[i, j] * m2[i, j]
    return res

def matrix_function_naive(m1, func):
    # Applique une fonction à tous les éléments : double boucle
    r, c = m1.shape
    res = alloc_matrix(r, c)
    for i in range(r):
        for j in range(c):
            res[i, j] = func(m1[i, j])
    return res

def matrix_transpose_naive(m):
    # Transposition d'une matrice : double boucle
    r, c = m.shape
    res = alloc_matrix(c, r)
    for i in range(r):
        for j in range(c):
            res[j, i] = m[i, j]
    return res

def matrix_scalar_naive(m, s):
    # Multiplication par un scalaire : double boucle
    r, c = m.shape
    res = alloc_matrix(r, c)
    for i in range(r):
        for j in range(c):
            res[i, j] = m[i, j] * s
    return res

def matrix_memcpy_naive(dest, src):
    # Copie élément par élément de src dans dest : double boucle
    r, c = src.shape
    for i in range(r):
        for j in range(c):
            dest[i, j] = src[i, j]


# %% [markdown]
# ### 3.2. Version Numpy
# (Utilise les opérations vectorisées de NumPy pour une exécution rapide)

# %%
def matrix_dot_numpy(m1, m2):
    return np.dot(m1, m2)

def matrix_sum_numpy(m1, m2):
    return m1 + m2

def matrix_minus_numpy(m1, m2):
    return m1 - m2

def hadamard_product_numpy(m1, m2):
    return m1 * m2

def matrix_function_numpy(m1, func):
    # On suppose que func est déjà vectorisée
    return func(m1)

def matrix_transpose_numpy(m):
    return m.T

def matrix_scalar_numpy(m, s):
    return m * s

def matrix_memcpy_numpy(dest, src):
    np.copyto(dest, src)


# %% [markdown]
# ---
# ## 4. Réseau de Neurones

# %%
class Layer:
    def __init__(self, layer_number, number_of_neurons, nneurons_previous_layer, minibatch_size):
        self.number_of_neurons = number_of_neurons
        self.minibatch_size = minibatch_size
        # activations et z de shape (nneurons, batch_size)
        self.activations = alloc_matrix(number_of_neurons, minibatch_size)
        self.z           = alloc_matrix(number_of_neurons, minibatch_size)
        self.delta       = alloc_matrix(number_of_neurons, minibatch_size)
        
        # weights de shape (nneurons, nneurons_previous_layer)
        self.weights     = alloc_matrix(number_of_neurons, nneurons_previous_layer)
        # biases de shape (nneurons, 1)
        self.biases      = alloc_matrix(number_of_neurons, 1)
        
        if layer_number > 0:
            self.init_weight(nneurons_previous_layer)
    
    def init_weight(self, nneurons_prev):
        # Initialisation vectorisée
        sigma = init_sigma(nneurons_prev)
        r, c = self.weights.shape
        self.weights = np.random.normal(0.0, sigma, size=(r, c))

class ANN:
    def __init__(self, alpha, minibatch_size, number_of_layers, nneurons_per_layer):
        self.alpha = alpha
        self.minibatch_size = minibatch_size
        self.number_of_layers = number_of_layers
        self.layers = []
        for i in range(number_of_layers):
            if i == 0:
                self.layers.append(
                    Layer(i, nneurons_per_layer[i], 
                          nneurons_per_layer[i],  
                          minibatch_size)
                )
            else:
                self.layers.append(
                    Layer(i, nneurons_per_layer[i],
                          nneurons_per_layer[i-1],
                          minibatch_size)
                )

def set_input(nn, input_matrix):
    # Utilise la version numpy pour la rapidité
    matrix_memcpy_numpy(nn.layers[0].activations, input_matrix)

def forward(nn, activation_function):
    for l in range(1, nn.number_of_layers):
        layer_l = nn.layers[l]
        layer_prev = nn.layers[l-1]
        z1 = matrix_dot_numpy(layer_l.weights, layer_prev.activations)
        ones = np.ones((1, nn.minibatch_size), dtype=np.float64)
        z2 = matrix_dot_numpy(layer_l.biases, ones)
        layer_l.z = matrix_sum_numpy(z1, z2)
        layer_l.activations = matrix_function_numpy(layer_l.z, activation_function)

def backward(nn, y, derivative_actfunct):
    L = nn.number_of_layers - 1
    layer_L = nn.layers[L]
    tmp = matrix_minus_numpy(layer_L.activations, y)
    dfzL = matrix_function_numpy(layer_L.z, derivative_actfunct)
    layer_L.delta = hadamard_product_numpy(tmp, dfzL)
    for l in range(L, 1, -1):
        layer_l = nn.layers[l]
        layer_lm1 = nn.layers[l-1]
        w_l_transp = matrix_transpose_numpy(layer_l.weights)
        delta_tmp = matrix_dot_numpy(w_l_transp, layer_l.delta)
        dfz = matrix_function_numpy(layer_lm1.z, derivative_actfunct)
        layer_lm1.delta = hadamard_product_numpy(delta_tmp, dfz)
    for l in range(1, nn.number_of_layers):
        layer_l = nn.layers[l]
        layer_lm1 = nn.layers[l-1]
        a_lm1_transp = matrix_transpose_numpy(layer_lm1.activations)
        w1 = matrix_dot_numpy(layer_l.delta, a_lm1_transp)
        w1 = matrix_scalar_numpy(w1, nn.alpha / nn.minibatch_size)
        layer_l.weights = matrix_minus_numpy(layer_l.weights, w1)
        ones = np.ones((nn.minibatch_size, 1), dtype=np.float64)
        b1 = matrix_dot_numpy(layer_l.delta, ones)
        b1 = matrix_scalar_numpy(b1, nn.alpha / nn.minibatch_size)
        layer_l.biases = matrix_minus_numpy(layer_l.biases, b1)


# %% [markdown]
# ---
# ## 5. Fonctions d'entraînement
# (Les versions Naïve et Numpy sont toutes deux fournies pour comparer les performances)

# %% [markdown]
# ### 5.1. Version Naïve

# %%
def populate_naive(x, y, minibatch_idx, train_img, train_label):
    """
    Remplit les matrices x et y avec le mini-batch de manière naïve (boucles explicites).
    x -> shape (784, minibatch_size)
    y -> shape (10, minibatch_size)
    """
    batch_size = len(minibatch_idx)
    for col, idx in enumerate(minibatch_idx):
        for row in range(784):
            x[row, col] = train_img[idx, row] / 255.0
        for row in range(10):
            y[row, col] = 0.0
        true_label = train_label[idx]
        y[true_label, col] = 1.0

def accuracy_naive(nn, test_img, test_label, minibatch_size):
    """
    Calcule l'accuracy en pourcentage sur l'ensemble des données test en mode naïf.
    On itère par mini-batch et utilise des boucles explicites pour déterminer les prédictions.
    """
    ntest = test_img.shape[0]
    good = 0
    idxs = zero_to_n(ntest)
    x = alloc_matrix(784, minibatch_size)
    y = alloc_matrix(10, minibatch_size)
    nbatches = (ntest // minibatch_size) * minibatch_size
    for i in range(0, nbatches, minibatch_size):
        batch_indices = idxs[i:i+minibatch_size]
        populate_naive(x, y, batch_indices, test_img, test_label)
        set_input(nn, x)
        forward(nn, sigmoid)
        last_activ = nn.layers[-1].activations
        for col in range(minibatch_size):
            max_val = -1e9
            max_idx = 0
            for row in range(10):
                val = last_activ[row, col]
                if val > max_val:
                    max_val = val
                    max_idx = row
            if max_idx == test_label[i + col]:
                good += 1
    return (100.0 * good) / nbatches

def cross_entropy_naive(y_pred, y_true, eps=1e-12):
    """
    Calcule la cross-entropy moyenne pour un mini-batch en mode naïf (boucles explicites).
    """
    batch_size = y_true.shape[1]
    ce_sum = 0.0
    for col in range(batch_size):
        for row in range(10):
            val_pred = y_pred[row, col]
            val_true = y_true[row, col]
            clipped = max(min(val_pred, 1.0 - eps), eps)
            if val_true > 0.0:
                ce_sum -= math.log(clipped)
    return ce_sum / batch_size


# %% [markdown]
# ### 5.2. Version Numpy

# %%
def populate_numpy(x, y, minibatch_idx, train_img, train_label):
    """
    Remplit les matrices x et y avec le mini-batch en utilisant des opérations vectorisées.
    x -> shape (784, minibatch_size)
    y -> shape (10, minibatch_size)
    """
    x_batch = train_img[minibatch_idx].astype(np.float64) / 255.0
    matrix_memcpy_numpy(x, x_batch.T)  # Transpose pour avoir (784, batch_size)
    y.fill(0.0)
    indices = train_label[minibatch_idx]
    y[indices, np.arange(len(minibatch_idx))] = 1.0

def accuracy_numpy(nn, test_img, test_label, minibatch_size):
    """
    Compute the accuracy (%) on the test set using fully vectorized NumPy operations.
    Processes the test set in mini-batches.
    """
    ntest = test_img.shape[0]
    nbatches = (ntest // minibatch_size) * minibatch_size
    correct = 0
    for i in range(0, nbatches, minibatch_size):
        batch_indices = np.arange(i, i + minibatch_size)
        x = test_img[batch_indices].T.astype(np.float64) / 255.0
        set_input(nn, x)
        forward(nn, sigmoid)
        preds = np.argmax(nn.layers[-1].activations, axis=0)
        correct += np.sum(preds == test_label[batch_indices])
    return (100.0 * correct) / nbatches

def cross_entropy_numpy(y_pred, y_true, eps=1e-12):
    """
    Compute cross-entropy error for a mini-batch using vectorized NumPy operations.
    """
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]


# %% [markdown]
# ---
# ## 6. Exécution Principale

# %%
DATA_PATH = "DATA"

# 6.1) Lecture des données MNIST
train_img = read_images(os.path.join(DATA_PATH, "train-images.idx3-ubyte"))
train_label = read_labels(os.path.join(DATA_PATH, "train-labels.idx1-ubyte"))
test_img = read_images(os.path.join(DATA_PATH, "t10k-images.idx3-ubyte"))
test_label = read_labels(os.path.join(DATA_PATH, "t10k-labels.idx1-ubyte"))

datasize = train_img.shape[0]
ntest = test_img.shape[0]

alpha = 0.05
minibatch_size = 16
number_of_layers = 3
nneurons_per_layer = [784, 30, 10]  # 28*28 = 784
nn = ANN(alpha, minibatch_size, number_of_layers, nneurons_per_layer)

shuffled_idx = zero_to_n(datasize)
x = alloc_matrix(784, minibatch_size)
y = alloc_matrix(10, minibatch_size)

# Using the numpy version for fast execution:
acc_start = accuracy_numpy(nn, test_img, test_label, minibatch_size)
print("Starting accuracy:", acc_start)

NEPOCHS = 20
for epoch in range(NEPOCHS):
    print(f"\nEPOCH: {epoch}")
    shuffle(shuffled_idx, datasize)
    nbatches = (datasize // minibatch_size) * minibatch_size
    batch_iter = range(0, nbatches, minibatch_size)
    ce_total = 0.0
    n_train_batches = 0
    acc = accuracy_numpy(nn, test_img, test_label, minibatch_size)
    desc = f'Epoch {epoch} - Acc: {acc:.2f}%'
    print(Fore.GREEN + desc + Style.RESET_ALL)
    for i in tqdm(batch_iter, desc=desc):
        batch_indices = shuffled_idx[i : i + minibatch_size]
        # To test the numpy version, call populate_numpy.
        populate_numpy(x, y, batch_indices, train_img, train_label)
        set_input(nn, x)
        forward(nn, sigmoid)
        y_pred = nn.layers[-1].activations  
        ce_batch = cross_entropy_numpy(y_pred, y)
        ce_total += ce_batch
        n_train_batches += 1
        backward(nn, y, dsigmoid)
    ce_mean = ce_total / n_train_batches
    acc = accuracy_numpy(nn, test_img, test_label, minibatch_size)
    desc = f'Epoch {epoch} - Acc: {acc:.2f}%, CE: {ce_mean:.4f}'
    print(Fore.GREEN + desc + Style.RESET_ALL)
