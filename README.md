# Car Crash

*L’efficacité exceptionnelle des réseaux de neurones les rend intéressant dans de nombreux domaines, en particulier celui de la conduite autonome. Mais est-ce bien raisonnable de leur faire autant confiance ?*

---

**Partie I.** Classification des panneaux de signalisation

**Partie II.** Tromper le réseau : comment hacker un panneau STOP ?

**Partie III.** Des pistes pour prémunir contre de telles attaques


## Plan détaillé

### Partie I.A (Prise en main des outils : entraînements sur MNIST)

- Étudier le fonctionnement d'un réseau de neurones basique (***MLP***).
- Mettre en place un *MLP* de classification des chiffres manuscrits de MNIST, sans framework en Python
- Même chose en utilisant *TensorFlow*.
- Étudier le fonctionnement des réseau neuronaux convolutifs (***CNN***).
- Mettre en place un CNN avec *TensorFlow* pour la reconaîssance de MNIST.

### Partie I.B (Classification des panneaux : GTSRB)

- Mettre en place un *MLP* avec *TensorFlow* pour la classification des panneaux (Réseau *A*).
- Mettre en place un *CNN* avec *TensorFlow* pour la classification des panneaux (Réseau *B*).
“sûr” de classification des panneaux de GTSRB qui servira de référence (Réseau *C*). (*Contacter Renault-Nissan ?*)
- Si possible : Faire fonctionner ces réseaux dans une application iOS pour avoir des résultats en temps réel.

### Partie II (Attaque des réseaux : *Dodging* puis *impersonating*)

- Sur les réseaux *A* et *B*, essayer une descente de gradient sur les pixels d'un panneau STOP reconnu par les réseaux jusqu'à les tromper. Étude mathématique de l'efficacité de cette descente.
- Étudier le fonctionnement des réseaux adversaires (***AN***) et adversaires génératifs (***GANs***).
- Essayer de mettre en place une attaque par *AN* sur les réseaux *A*, *B* et *C*.
- Étudier et comparer les résultats.

### Partie III (Renforcement des réseaux)
- Étudier les renforcements possibles des réseaux étudiés par exemple par des *GANs*.
- À compléter...
- Mettre en place ces stratégies, et nouvelle tentative de “hack” pour voir leur effet.
- Conclure

---

## Bibliographie

### Livres

- [**1**] **Neural Networks and Deep Learning** (2015), Michael A. Nielsen [[web]](http://neuralnetworksanddeeplearning.com)

> Très bonne introduction au DL, avec des applications directes en Python.

- [**2**] **Deep Learning** (2016), Ian Goodfellow, Yoshua Bengio & Aaron Courville [[web]](http://deeplearningbook.org)

> Livre exausthif sur le Deep Learning, une très grosse référence.

### Publications

- [**3**] **Deep Learning**, *Nature* (2015), Yann LeCun, Yoshua Bengio & Geoffrey Hinton [[pdf]](http://pages.cs.wisc.edu/~dyer/cs540/handouts/deep-learning-nature2015.pdf)

> Un “état de l'art” du DL par deux spécialistes du domaines dans la prestigieuse revue *Nature*.

- [**4**] **ImageNet Classification with Deep Convolutional Neural Networks** (2012), A. Krizhevsky et al. [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

> Cette publication explique comment un CNN, avec une architecture appelée AlexNet a pulvérisé le concours annuel de classification d'objets ImageNet (22000 catégories !) en 2012. Depuis, les CNNs sont les leaders de ce segment.
> Il y justifie l'intérêt de ReLU et de l'augmentation des données (rotation d'images, flips...), et l'intérêt des CNNs en général dans ce domaine.

- [**5**] **Generative adversarial nets** (2014), I. Goodfellow et al. [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

> Une publication d'importance capitale, qui a lancé l'étude des GANs. Cette technique permet d'utiliser les réseaux de neurones comme générateurs d'images.
> The analogy used in the paper is that the generative model is like “a team of counterfeiters, trying to produce and use fake currency” while the discriminative model is like “the police, trying to detect the counterfeit currency”. The generator is trying to fool the discriminator while the discriminator is trying to not get fooled by the generator. As the models train, both methods are improved until a point where the “counterfeits are indistinguishable from the genuine articles”.

- [**6**] **Explaining and Harnessing Adversarial Examples** (2015), I. Goodfellow et al. [[pdf]](https://arxiv.org/pdf/1412.6572.pdf)

> **Abstract:** Several machine learning models, including neural networks, consistently misclassify adversarial examples -- inputs formed by applying small but intentionally worst-case perturbations to examples from the dataset, such that the perturbed input results in the model outputting an incorrect answer with high confidence. Early attempts at explaining this phenomenon focused on nonlinearity and overfitting. We argue instead that the primary cause of neural networks' vulnerability to adversarial perturbation is their linear nature. This explanation is supported by new quantitative results while giving the first explanation of the most intriguing fact about them: their generalization across architectures and training sets. Moreover, this view yields a simple and fast method of generating adversarial examples. Using this approach to provide examples for adversarial training, we reduce the test set error of a maxout network on the MNIST dataset.

- [**7**] **Robust Physical-World Attacks on Machine Learning Models** (2017) [[pdf]](https://arxiv.org/pdf/1707.08945.pdf)

> Papier om est essayée l'attaque de panneaux STOP.

- [**8**] **Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition** (2016) [[pdf]](https://www.cs.cmu.edu/~sbhagava/papers/face-rec-ccs16.pdf)

> Comment tromper une reconnaissance faciale avec de simples lunettes ?

### Sources complémentaires

- [**_**] **Practical Black-Box Attacks against Machine Learning** (2017), I. Goodfellow et al.  [[pdf]](https://arxiv.org/pdf/1602.02697v4.pdf)

- [**_**] **NIPS 2016 Tutorial: Generative Adversarial Networks** [[pdf]](https://arxiv.org/pdf/1701.00160v4.pdf)


## Liens utiles

- [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb)
- [Arxiv Sanity Preserver (Recherche de publications)](http://www.arxiv-sanity.com)
- [Très bon glossaire](http://www.wildml.com/deep-learning-glossary/)

