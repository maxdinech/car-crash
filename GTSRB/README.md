# GTSRB

## Structure des données

Le dossier `data/Training/` correspond à la base de données officielle de GTSRB.
Il est organisé en 43 sous-dossier de catégories, chacun contenant des images au
format `.ppm`.

Le dossier `data/validation` contient les 3000 labels et images et validation
sélectionnées dans celles de GTSRB, sous forme de tableaux numpy. Les images ont
été normalisées au format 40x40 et sont disposibles dans différents profils de
couleurs : `rgb`, `grey` et `clahe`.

Le dossier `data/train` contient les autres images et labels de GTSRB, sous
forme de tableaux numpy. Les images ont été normalisées au format 40x40 et sont
disposibles dans différents profils de couleurs : `rgb`, `grey` et `clahe`. Des
versions `ext` sont disposibles pour chaque profil : elles correspondent à
l'augmentation par symétries des bases de données.


## Performances obtenues

Tous les entraînements sont réalisés avec 12 epochs, et une taille de paquet de
128, sur le même réseau.

La performance est à chaque fois mesurée sur la même base de validation.

### base `grey`

| **Epoch** |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  | 10  | 11  | 12  |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Prec.** |     |     |     |     |     |     |     |     |     |     |     |     |

### base `grey` étendue par symétries

| **Epoch** |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  | 10  | 11  | 12  |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Prec.** |     |     |     |     |     |     |     |     |     |     |     |     |

### base `grey` étendue par symétries et distorsions

| **Epoch** |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  | 10  | 11  | 12  |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Prec.** |     |     |     |     |     |     |     |     |     |     |     |     |

### base `clahe`

| **Epoch** |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  | 10  | 11  | 12  |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Prec.** |     |     |     |     |     |     |     |     |     |     |     |     |

### base `clahe` étendue par symétries

| **Epoch** |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  | 10  | 11  | 12  |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Prec.** |91.47|95.30|     |     |     |     |     |     |     |     |     |     |

### base `clahe` étendue par symétries et distorsions

| **Epoch** |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  | 10  | 11  | 12  |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Prec.** |     |     |     |     |     |     |     |     |     |     |     |     |