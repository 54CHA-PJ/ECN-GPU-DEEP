import math
import random
import time
import gzip
import struct
from tqdm import tqdm

# ----------------------------------------------------------
# FONCTIONS UTILITAIRES
# ----------------------------------------------------------

def zero_to_n(n, t):
    """
    Remplit la liste t avec les entiers 0..n-1
    """
    for i in range(n):
        t[i] = i

def shuffle(t, size, number_of_switch):
    """
    Mélange la liste t aléatoirement (équivalent de la fonction C).
    """
    zero_to_n(size, t)
    for _ in range(number_of_switch):
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        tmp = t[x]
        t[x] = t[y]
        t[y] = tmp

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1.0 - sigmoid(x))

def populate_minibatch(x, y, minibatch_idx, minibatch_size, img, img_size, label, label_size):
    """
    Remplit x et y avec les données correspondantes au minibatch.
    - x : tableau 1D de taille (img_size * minibatch_size)
    - y : tableau 1D de taille (label_size * minibatch_size)
    - minibatch_idx : liste d’indices pour ce minibatch
    - img : liste de toutes les images (chacune étant de taille img_size)
    - label : liste de tous les labels (taille label_size=10 pour MNIST)
    """
    for col in range(minibatch_size):
        # Charger les pixels
        for row in range(img_size):
            # Normalisation 0..1
            x[row * minibatch_size + col] = float(img[minibatch_idx[col]][row]) / 255.0

        # Mettre le vecteur Y à zéro
        for row in range(label_size):
            y[row * minibatch_size + col] = 0.0

        # Mettre la bonne classe à 1.0
        y[label[minibatch_idx[col]] * minibatch_size + col] = 1.0

# ----------------------------------------------------------
# GESTION MNIST
# ----------------------------------------------------------

def read_images(filename, out_size):
    """
    Lit les images MNIST (au format IDX) depuis un fichier compressé .gz
    et renvoie (images, size).

    - filename : chemin vers le fichier .gz (ex: 'DATA/train-images-idx3-ubyte.gz')
    - out_size : argument factice pour rester fidèle au code C (ignoré en Python)
    """
    with gzip.open(filename, 'rb') as f:
        # Lecture de l'en-tête (16 octets)
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))

        # Par convention MNIST, magic = 0x00000803
        # rows=28, cols=28 pour MNIST standard
        if magic != 2051:
            raise ValueError(f"Fichier d'images invalide (magic number={magic})")

        images = []
        for _ in range(num_images):
            # Lecture de 28*28 = 784 octets
            img_data = f.read(rows * cols)
            # On stocke chaque pixel (byte) dans un tableau
            # (chaque pixel est un int entre 0 et 255)
            images.append(list(img_data))

        return images, num_images

def read_labels(filename, out_size):
    """
    Lit les labels MNIST (au format IDX) depuis un fichier compressé .gz
    et renvoie (labels, size).

    - filename : chemin vers le fichier .gz (ex: 'DATA/train-labels-idx1-ubyte.gz')
    - out_size : argument factice pour rester fidèle au code C (ignoré en Python)
    """
    with gzip.open(filename, 'rb') as f:
        # Lecture de l'en-tête (8 octets)
        magic, num_labels = struct.unpack('>II', f.read(8))

        # Par convention MNIST, magic = 0x00000801
        if magic != 2049:
            raise ValueError(f"Fichier de labels invalide (magic number={magic})")

        labels_data = f.read(num_labels)
        # On stocke chaque label (byte) dans un tableau
        labels = list(labels_data)

        return labels, num_labels

# ----------------------------------------------------------
# MATRICES ET RÉSEAU DE NEURONES (SIMPLIFIÉ)
# ----------------------------------------------------------

class matrix_t:
    """
    Équivalent d’une struct matrix_t en C. On va stocker un tableau Python 1D.
    rows et cols sont juste des attributs informatifs.
    """
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        # On stocke dans un simple tableau 1D, 
        # comme dans le C on avait un malloc(...) d’une taille rows*cols
        self.m = [0.0]*(rows*cols)

def alloc_matrix(rows, cols):
    """
    Équivalent de alloc_matrix(rows, cols).
    """
    return matrix_t(rows, cols)

def destroy_matrix(mat):
    """
    Équivalent de destroy_matrix. En Python, on détruit simplement l’objet 
    en le déréférençant, mais on fournit la fonction pour coller au code C.
    """
    del mat

# ----------------------------------------------------------
# DÉFINITION DU RÉSEAU ann_t ET FONCTIONS CREATE_ANN, FORWARD, BACKWARD
# ----------------------------------------------------------

class layer_t:
    """
    Une couche du réseau, contenant:
    - activations : matrix_t
    - weights : matrix_t
    - biases : matrix_t
    etc.
    """
    def __init__(self, nb_neurons, minibatch_size):
        # Ici, on se contente de stocker les activations 
        # (dans le code C, on aurait plus d’infos, par ex. weights/bias).
        self.activations = alloc_matrix(nb_neurons, minibatch_size)

class ann_t:
    """
    Équivalent d’une structure ann_t en C,
    contenant par exemple : 
    - alpha
    - minibatch_size
    - number_of_layers
    - layers
    """
    def __init__(self, alpha, minibatch_size, number_of_layers, nneurons_per_layer):
        self.alpha = alpha
        self.minibatch_size = minibatch_size
        self.number_of_layers = number_of_layers

        # On crée la liste de couches
        self.layers = []
        for i in range(number_of_layers):
            nb_neurons = nneurons_per_layer[i]
            layer = layer_t(nb_neurons, minibatch_size)
            self.layers.append(layer)

def create_ann(alpha, minibatch_size, number_of_layers, nneurons_per_layer):
    """
    Équivalent de create_ann(...) dans le code C.
    """
    nn = ann_t(alpha, minibatch_size, number_of_layers, nneurons_per_layer)
    return nn

def forward(nn, activation_func):
    """
    Parcours avant du réseau (équivalent de forward).
    On suppose que la couche 0 a déjà ses activations remplies
    (nn->layers[0]->activations).
    Dans le code C, la logique de forward utiliserait les weights, 
    réaliserait la multiplication, etc. 
    Ici, on fait juste un placeholder pour la démonstration.
    """
    for layer_idx in range(1, nn.number_of_layers):
        in_layer = nn.layers[layer_idx - 1]
        out_layer = nn.layers[layer_idx]
        
        # TODO: Multiplication poids * activations + biais, puis activation.
        # Dans le code C : 
        # out_layer->activations = activation_func( W * in_layer->activations + b )
        #
        # Ici, en guise de placeholder, on recopie juste in_layer dans out_layer
        # avec l’activation non-linéaire.
        
        for r in range(out_layer.activations.rows):
            for c in range(out_layer.activations.cols):
                # index pour la matrice out_layer
                idx = r * out_layer.activations.cols + c
                # index dans la couche précédente 
                idx_in = r % in_layer.activations.rows
                val_in = in_layer.activations.m[idx_in * in_layer.activations.cols + c]
                out_layer.activations.m[idx] = activation_func(val_in)

def backward(nn, out, dactivation_func):
    """
    Rétropropagation (équivalent de backward).
    Dans le code C, on ferait la mise à jour des gradients et des poids
    en fonction de l’écart (out_layer - out) * dactivation_func(...).
    Ici, c’est un simple placeholder.
    """
    # TODO: Implémenter la vraie rétropropagation : 
    #      - Calcul de l’erreur de sortie
    #      - Propagation en arrière
    #      - Mise à jour des poids
    pass

# ----------------------------------------------------------
# FONCTION ACCURACY (ÉVALUATION)
# ----------------------------------------------------------

def accuracy(test_img, test_label, datasize, minibatch_size, nn):
    """
    Calcule la précision en traitant test_img/test_label par minibatch,
    en utilisant forward(nn, sigmoid). On récupère l’index du neurone 
    de sortie le plus grand, et on compare à test_label.
    """
    good = 0
    idx = [0]*datasize
    zero_to_n(datasize, idx)

    x = [0.0]*(28*28*minibatch_size)
    y = [0.0]*(10*minibatch_size)

    # Boucle sur l’ensemble du dataset, minibatch par minibatch
    for i in range(0, datasize - minibatch_size, minibatch_size):
        # Remplit x et y
        populate_minibatch(x, y, idx[i:i+minibatch_size], 
                           minibatch_size, 
                           test_img, 28*28, 
                           test_label, 10)

        # Copie dans les activations de la première couche
        for k in range(28*28*minibatch_size):
            nn.layers[0].activations.m[k] = x[k]

        # Forward
        forward(nn, sigmoid)

        # Récupération des activations de la dernière couche
        out_layer = nn.layers[nn.number_of_layers - 1].activations
        for col in range(minibatch_size):
            idx_training_data = col + i
            maxval = -999999.0
            idx_max = 0
            for row in range(10):
                idx_out = row * minibatch_size + col
                if out_layer.m[idx_out] > maxval:
                    maxval = out_layer.m[idx_out]
                    idx_max = row
            # Compare au label effectif
            if idx_max == test_label[idx_training_data]:
                good += 1

    ntests = (datasize // minibatch_size) * minibatch_size
    return 100.0 * float(good) / float(ntests)

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    random.seed(int(time.time()))

    # Lecture des images / labels depuis le dossier DATA
    train_img, datasize = read_images("DATA/train-images-idx3-ubyte.gz", 0)
    train_label, datasize = read_labels("DATA/train-labels-idx1-ubyte.gz", 0)

    test_img, ntest = read_images("DATA/t10k-images-idx3-ubyte.gz", 0)
    test_label, ntest = read_labels("DATA/t10k-labels-idx1-ubyte.gz", 0)

    alpha = 0.05
    minibatch_size = 16
    number_of_layers = 3
    nneurons_per_layer = [28*28, 30, 10]

    # Création du réseau
    nn = create_ann(alpha, minibatch_size, number_of_layers, nneurons_per_layer)

    print("starting accuracy", accuracy(test_img, test_label, ntest, minibatch_size, nn))

    # Allocation d’un buffer pour x, y et shuffled_idx
    shuffled_idx = [0]*datasize
    x = [0.0]*(28*28*minibatch_size)
    y = [0.0]*(10*minibatch_size)
    out = alloc_matrix(10, minibatch_size)

    # Boucle d’entraînement
    for epoch in range(40):
        print(f"start learning epoch {epoch}")
        
        # Mélanger les indices
        shuffle(shuffled_idx, datasize, datasize)

        for i in range(0, datasize - minibatch_size, minibatch_size):
            # Préparer le minibatch
            populate_minibatch(x, y, shuffled_idx[i:i+minibatch_size], 
                               minibatch_size, 
                               train_img, 28*28, 
                               train_label, 10)

            # Copier x dans la couche 0
            for k in range(28*28*minibatch_size):
                nn.layers[0].activations.m[k] = x[k]
            
            # Forward
            forward(nn, sigmoid)

            # Copier y dans out->m (équivalent de memcpy)
            for k in range(10*minibatch_size):
                out.m[k] = y[k]

            # Backward
            backward(nn, out, dsigmoid)

        print(f"epoch {epoch} accuracy {accuracy(test_img, test_label, ntest, minibatch_size, nn)}")

    # Libérations mémoire
    destroy_matrix(out)

if __name__ == "__main__":
    main()
