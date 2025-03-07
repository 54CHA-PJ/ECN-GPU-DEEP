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
import struct
from tqdm import tqdm
from colorama import Fore, Style
import math
import os


# %% [markdown]
# ---
# ## 1. Lecture des fichiers MNIST

# %%
def make_uint32(byte_array):
    # Recompose un entier 32 bits à partir de 4 octets 
    # de poids fort à poids faible
    return ((byte_array[0] << 24) 
          | (byte_array[1] << 16) 
          | (byte_array[2] <<  8) 
          | (byte_array[3] <<  0))

def read_labels(filename):
    with open(filename, 'rb') as f:
        # Magic number (4 octets) – non utilisé ici
        _ = f.read(4)
        
        # Nombre d'étiquettes
        n_bytes = f.read(4)
        n = make_uint32(n_bytes)

        # Lecture des labels
        labels = np.frombuffer(f.read(n), dtype=np.uint8)
    return labels

def read_images(filename):
    with open(filename, 'rb') as f:
        # Magic number (4 octets) – non utilisé ici
        _ = f.read(4)
        
        # Nombre d'images
        n_bytes = f.read(4)
        n = make_uint32(n_bytes)

        # Nombre de lignes et de colonnes (ici 28x28)
        row_bytes = f.read(4)
        col_bytes = f.read(4)
        rows = make_uint32(row_bytes)
        cols = make_uint32(col_bytes)
        
        # Lecture des pixels (chacun sur 1 octet)
        images_raw = f.read(n * rows * cols)
        # On crée un tableau NumPy de shape (n, rows*cols)
        # pour être cohérent avec le C qui stockait chaque image 
        # en un tableau 1D de 28*28
        images = np.frombuffer(images_raw, dtype=np.uint8)
        images = images.reshape(n, rows * cols)
    return images


# %% [markdown]
# ---
# ## 2. Fonctions utilitaires

# %%
def zero_to_n(n):
    """
    Crée un tableau [0, 1, 2, ..., n-1].
    """
    return np.arange(n, dtype=np.uint32)

def shuffle(t, number_of_switch):
    """
    Mélange un tableau t aléatoirement, 
    en réalisant 'number_of_switch' échanges.
    """
    # Dans le code C, le mélange se fait en échangeant
    # aléatoirement number_of_switch fois.
    size = len(t)
    for _ in range(number_of_switch):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        tmp = t[x]
        t[x] = t[y]
        t[y] = tmp

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
    # Dérivée de sigmoid
    return sigmoid(x) * (1.0 - sigmoid(x))

def cross_entropy_error(y_pred, y_true, eps=1e-12):
    """
    Calcule la cross-entropy moyenne pour un mini-batch.
    y_pred et y_true sont de shape (10, batch_size).
    """
    # On s'assure d'éviter log(0)
    clipped = np.clip(y_pred, eps, 1.0 - eps)
    batch_size = y_true.shape[1]
    # somme de - y_true * log(y_pred) sur toutes les classes, 
    # puis moyenne sur le batch
    ce = - np.sum(y_true * np.log(clipped)) / batch_size
    return ce


# %% [markdown]
# ---
# ## 3. Réseau de neurones

# %%
def alloc_matrix(rows, columns):
    # En C, on faisait un malloc puis un tableau 1D rows*columns.
    # En Python, on retournera un np.ndarray de shape (rows, columns).
    return np.zeros((rows, columns), dtype=np.float64)

def matrix_dot(m1, m2):
    # m1 shape = (rows1, col1)
    # m2 shape = (rows2, col2)
    # assert col1 == rows2
    return np.dot(m1, m2)

def matrix_sum(m1, m2):
    return m1 + m2

def matrix_minus(m1, m2):
    return m1 - m2

def hadamard_product(m1, m2):
    return m1 * m2

def matrix_function(m1, func):
    # Applique f à chaque élément
    return func(m1)

def matrix_transpose(m):
    return m.T

def matrix_scalar(m, s):
    return m * s

def matrix_memcpy(dest, src):
    # recopie le contenu de src dans dest, supposé de la même shape
    np.copyto(dest, src)

#
# Définitions des couches et du réseau
#
class Layer:
    def __init__(self, layer_number, number_of_neurons, nneurons_previous_layer, minibatch_size):
        self.number_of_neurons = number_of_neurons
        self.minibatch_size = minibatch_size
        # Matrices
        # activations et z de shape (nneurons, batch_size)
        self.activations = alloc_matrix(number_of_neurons, minibatch_size)
        self.z           = alloc_matrix(number_of_neurons, minibatch_size)
        self.delta       = alloc_matrix(number_of_neurons, minibatch_size)
        
        # weights de shape (nneurons, nneurons_previous_layer)
        self.weights     = alloc_matrix(number_of_neurons, nneurons_previous_layer)
        # biases de shape (nneurons, 1)
        self.biases      = alloc_matrix(number_of_neurons, 1)
        
        # Initialisation des poids si ce n'est pas la couche d'entrée
        if layer_number > 0:
            self.init_weight(nneurons_previous_layer)
    
    def init_weight(self, nneurons_prev):
        # Equivalent de init_weight() en C (Xavier init approchée, via normalRand)
        # On utilise np.random.normal(0, 1/sqrt(nneurons_prev), ...)
        sigma = 1.0 / math.sqrt(nneurons_prev)
        self.weights[:] = np.random.normal(0.0, sigma, size=self.weights.shape)
        
class ANN:
    def __init__(self, alpha, minibatch_size, number_of_layers, nneurons_per_layer):
        self.alpha = alpha
        self.minibatch_size = minibatch_size
        self.number_of_layers = number_of_layers
        
        # Création des couches
        self.layers = []
        for i in range(number_of_layers):
            if i == 0:
                # Couche d'entrée
                self.layers.append(
                    Layer(i, nneurons_per_layer[i], 
                          nneurons_per_layer[i],  # la taille "précédente" n'a pas d'impact pour la couche d'entrée
                          minibatch_size)
                )
            else:
                # Couches cachées / sortie
                self.layers.append(
                    Layer(i, nneurons_per_layer[i],
                          nneurons_per_layer[i-1],
                          minibatch_size)
                )

def set_input(nn, input_matrix):
    # Recopie input_matrix dans la "couche 0" (couche d'entrée)
    matrix_memcpy(nn.layers[0].activations, input_matrix)

def forward(nn, activation_function):
    # On parcourt les couches de 1 à L-1 en faisant:
    # z^l = w^l . a^(l-1) + b^l
    # a^l = f(z^l)
    for l in range(1, nn.number_of_layers):
        layer_l = nn.layers[l]
        layer_prev = nn.layers[l - 1]
        
        # z1 = w^l dot a^(l-1)
        z1 = matrix_dot(layer_l.weights, layer_prev.activations)
        
        # z2 = b^l dot 1, où 1 est un vecteur-ligne de taille minibatch 
        # => (nneurons, 1) x (1, minibatch_size) = (nneurons, minibatch_size)
        ones = np.ones((1, nn.minibatch_size), dtype=np.float64)
        z2 = matrix_dot(layer_l.biases, ones)
        
        # z^l
        layer_l.z = matrix_sum(z1, z2)
        
        # a^l
        layer_l.activations = matrix_function(layer_l.z, activation_function)

def backward(nn, y, derivative_actfunct):
    # L = index de la dernière couche
    L = nn.number_of_layers - 1

    # 1) Calcul de delta^L = (a^L - y) hadamard f'(z^L)
    layer_L = nn.layers[L]
    tmp = matrix_minus(layer_L.activations, y)  # a^L - y
    dfzL = matrix_function(layer_L.z, derivative_actfunct)  # f'(z^L)
    layer_L.delta = hadamard_product(tmp, dfzL)

    # 2) Rétropropagation pour l= L..2
    #    delta^(l-1) = (w^l)^T dot delta^l hadamard f'(z^(l-1))
    for l in range(L, 1, -1):
        layer_l     = nn.layers[l]
        layer_lm1   = nn.layers[l - 1]  # l-1
        w_l_transp  = matrix_transpose(layer_l.weights)
        delta_tmp   = matrix_dot(w_l_transp, layer_l.delta)
        dfz         = matrix_function(layer_lm1.z, derivative_actfunct)
        layer_lm1.delta = hadamard_product(delta_tmp, dfz)

    # 3) Mise à jour poids & biais
    #    w^l = w^l - ( alpha / m ) * ( delta^l dot (a^(l-1))^T )
    #    b^l = b^l - ( alpha / m ) * ( delta^l dot 1 )
    for l in range(1, nn.number_of_layers):
        layer_l     = nn.layers[l]
        layer_lm1   = nn.layers[l - 1]
        
        a_lm1_transp = matrix_transpose(layer_lm1.activations)
        
        # gradient w
        w1 = matrix_dot(layer_l.delta, a_lm1_transp)
        w1 = matrix_scalar(w1, nn.alpha / nn.minibatch_size)
        layer_l.weights = matrix_minus(layer_l.weights, w1)
        
        # gradient b
        ones = np.ones((nn.minibatch_size, 1), dtype=np.float64)
        b1 = matrix_dot(layer_l.delta, ones)  # shape (nneurons, 1)
        b1 = matrix_scalar(b1, nn.alpha / nn.minibatch_size)
        layer_l.biases = matrix_minus(layer_l.biases, b1)


# %% [markdown]
# ---
# ## 4. Fonctions utilisaires pour l'entraînement

# %%
def populate_minibatch(x, y, minibatch_idx, train_img, train_label):
    """
    x -> shape (784, minibatch_size)
    y -> shape (10,   minibatch_size)
    minibatch_idx -> indices des exemples à mettre dans le batch
    train_img et train_label sont les données entières
    """
    batch_size = len(minibatch_idx)
    for col, idx in enumerate(minibatch_idx):
        # Remplir x (784, batch_size)
        # Normaliser sur 255
        x[:, col] = train_img[idx] / 255.0

        # Remplir y (10, batch_size)
        y[:, col] = 0.0
        true_label = train_label[idx]
        y[true_label, col] = 1.0

def accuracy(nn, test_img, test_label, minibatch_size):
    """
    Calcule l'accuracy en pourcentage sur l'ensemble des données test.
    On effectue des mini-batches de taille 'minibatch_size'.
    """
    ntest = test_img.shape[0]
    good = 0
    # Indices
    idxs = zero_to_n(ntest)
    
    # Buffers pour x,y
    x = alloc_matrix(784, minibatch_size)
    y = alloc_matrix(10,   minibatch_size)
    
    # On itère par minibatch
    nbatches = (ntest // minibatch_size) * minibatch_size
    for i in range(0, nbatches, minibatch_size):
        batch_indices = idxs[i:i+minibatch_size]
        
        populate_minibatch(x, y, batch_indices, test_img, test_label)
        # On met x dans la couche d'entrée
        set_input(nn, x)
        # Forward
        forward(nn, sigmoid)
        
        # Prédictions
        # nn.layers[-1].activations -> shape (10, minibatch_size)
        # On récupère l'indice de la classe la plus probable
        predictions = np.argmax(nn.layers[-1].activations, axis=0)  # dimension= batch_size
        # Comparaison
        good_batch = np.sum(predictions == test_label[i:i+minibatch_size])
        good += good_batch
    
    return (100.0 * good) / nbatches


# %% [markdown]
# ---
# ## 5. Execution Principale

# %%
DATA_PATH = "DATA"

# 6.1) Lecture des données MNIST
train_img = read_images(DATA_PATH + "/train-images.idx3-ubyte")
train_label = read_labels(DATA_PATH + "/train-labels.idx1-ubyte")
test_img = read_images(DATA_PATH + "/t10k-images.idx3-ubyte")
test_label = read_labels(DATA_PATH + "/t10k-labels.idx1-ubyte")

datasize = train_img.shape[0]
ntest    = test_img.shape[0]

# %%
# 6.2) Création du réseau de neurones
alpha = 0.05
minibatch_size = 16
number_of_layers = 3
nneurons_per_layer = [784, 30, 10]  # 28*28 = 784
nn = ANN(alpha, minibatch_size, number_of_layers, nneurons_per_layer)

# %%
# 6.3) Buffers utiles pour l'entraînement
shuffled_idx = zero_to_n(datasize)
x = alloc_matrix(784, minibatch_size)
y = alloc_matrix(10,   minibatch_size)

# %%
acc_start = accuracy(nn, test_img, test_label, minibatch_size)
print("Starting accuracy:", acc_start)

NEPOCHS = 20
# 6.4) Boucle d'entraînement (10 itérations)
# On calcule par exemple l'accuracy et la cross-entropy à chaque epoch.
for epoch in range(NEPOCHS):
    # On mélange les indices
    shuffle(shuffled_idx, datasize)
    
    # On va parcourir l'ensemble du jeu d'entraînement par minibatch
    nbatches = (datasize // minibatch_size) * minibatch_size
    batch_iter = range(0, nbatches, minibatch_size)
    
    # Accumulateur pour la cross-entropy 
    ce_total = 0.0
    n_train_batches = 0
    
    # Calculer l'accuracy actuelle
    acc = accuracy(nn, test_img, test_label, minibatch_size)
    desc = f'Epoch {epoch} - Acc: {acc:.2f}%'
    
    for i in tqdm(batch_iter, desc=desc):
        batch_indices = shuffled_idx[i : i + minibatch_size]
        populate_minibatch(x, y, batch_indices, train_img, train_label)
        
        set_input(nn, x)
        forward(nn, sigmoid)
        
        # Récupérer la prédiction pour calculer la cross-entropy
        y_pred = nn.layers[-1].activations  
        ce_batch = cross_entropy_error(y_pred, y)
        ce_total += ce_batch
        n_train_batches += 1

        backward(nn, y, dsigmoid)

    # Moyenne cross-entropy de l'epoch
    ce_mean = ce_total / n_train_batches
    acc = accuracy(nn, test_img, test_label, minibatch_size)
    
    desc = f'Epoch {epoch} - Acc: {acc:.2f}%, CE: {ce_mean:.4f}'

