# Read me
## Separation du dataset
```bash
python3 separate.py <dataset.csv>
```
## Entrainement du programme
```bash
python3 train.py <dataset.csv> <network.json>
```
## Guide de création d'un fichier `network.json`
Ce fichier JSON est utilisé pour définir la structure d'un réseau de neurones artificiels ainsi que les paramètres d'entraînement. Plusieurs réseaux peuvent être définis dans le fichier, mais seul celui spécifié dans la section `model_fit` sera pris en compte.

---

## Structure du fichier `network.json`

Voici la structure générale du fichier :

```json
{
    "network_1": {
        "input_layer": {
            "activation": "default"
        },
        "hidden_layer_1": {
            "neurons": 20,
            "activation": "sigmoid",
            "weights_init": "heUniform"
        },
        "hidden_layer_2": {
            "neurons": 7,
            "activation": "sigmoid",
            "weights_init": "heUniform"
        },
        "output_layer": {
            "activation": "softmax",
            "weights_init": "heUniform"
        }
    },
    "model_fit": {
        "network": "network_1",
        "loss": "binaryCrossEntropy",
        "learning_rate": 0.08,
        "batch_size": 32,
        "epochs": 100
    }
}
```

---

## Détails des éléments

### 1. Définition du réseau

Un réseau est défini par plusieurs couches :

- **`input_layer`** :
  - `activation`: Spécifie la fonction d'activation de la couche d'entrée (valeur par défaut "default").

- **Couches cachées (`hidden_layer_X`)** :
  - `neurons`: Nombre de neurones dans la couche.
  - `activation`: Fonction d'activation (ex. "sigmoid", "relu", etc.).
  - `weights_init`: Méthode d'initialisation des poids (ex. "heUniform").

- **`output_layer`** :
  - `activation`: Fonction d'activation (ex. "softmax").
  - `weights_init`: Méthode d'initialisation des poids.

**Remarque :** Le nombre de couches cachées est illimité et elles doivent être nommées selon la convention `hidden_layer_1`, `hidden_layer_2`, etc.

### 2. Paramètres d'entraînement (`model_fit`)

Cette section définit les paramètres d'entraînement du modèle. Les champs disponibles sont :

- `network`: Le nom du réseau à utiliser.
- `loss`: Fonction de coût (doit être **obligatoirement** `binaryCrossEntropy`).
- `learning_rate`: Taux d'apprentissage (ex. 0.08).
- `batch_size`: Taille des lots pour l'entraînement (ex. 32).
- `epochs`: Nombre d'époques d'entraînement (ex. 100).

---

## Notes importantes

1. Seul le réseau mentionné dans la section `model_fit` sera pris en compte pour l'entraînement.
2. Il est possible de définir plusieurs réseaux, mais un seul peut être utilisé à la fois.
3. La fonction de coût supportée est uniquement `binaryCrossEntropy`.
4. Le nombre de couches cachées est illimité, mais elles doivent être nommées de manière consécutive.

---

## Exemple d'un autre réseau

```json
{
    "network_2": {
        "input_layer": {
            "activation": "default"
        },
        "hidden_layer_1": {
            "neurons": 15,
            "activation": "relu",
            "weights_init": "xavier"
        },
        "hidden_layer_2": {
            "neurons": 10,
            "activation": "tanh",
            "weights_init": "heNormal"
        },
        "output_layer": {
            "activation": "softmax",
            "weights_init": "xavier"
        }
    }
}
```

