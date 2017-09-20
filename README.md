# Car Crash

*L’efficacité exceptionnelle des réseaux de neurones les rend intéressant dans de nombreux domaines, en particulier celui de la conduite autonome. Mais est-ce bien raisonnable de leur faire autant confiance ?*

## Thème - Milieux : interactions, interfaces, homogénéité, ruptures

On cherche à donner à la voiture une forme d'intelligence qui en fasse un agent quasi-autonome au sein de son **milieu** : identification des autres agents, prédiction de leurs trajectoires et **interactions**, identification et reconaîssance des informations routières (panneaux, marquages au sol, voies de circulation).

On utilise principalement pour ces tâches les réseaux de neurones, qui sont aujourd'hui la meilleure **interface** de compréhension d'images (Computer Vision). Ils consistent à séparer un espace R^n finement : les frontières de classifications -> **ruptures**

---

**Partie I.** Classification des panneaux de signalisation

**Partie II.** Tromper le réseau : comment hacker un panneau STOP ?

**Partie III.** Des pistes pour se prémunir contre de telles attaques


## Plan détaillé

### Partie I.A (Prise en main des outils : entraînements sur *MNIST*)

- [x] Étudier le fonctionnement d'un réseau de neurones basique (***MLP***).
- [x] Mettre en place un *MLP* de classification des chiffres manuscrits de *MNIST*, sans framework en Python.
- [x] Même chose en utilisant *Keras*.
- [x] Étudier le fonctionnement des réseau neuronaux convolutifs (***CNN***).
- [x] Mettre en place un CNN avec *Keras* pour la reconaîssance de *MNIST*.

### Partie I.B (Classification des panneaux : *GTSRB*)

- [x] Mettre en place un *CNN* avec *Keras* pour la classification des panneaux (Réseau *A*).
- [ ] Trouver un réseau de classification des panneaux utilisé dans des voitures du commerce (Réseau *B*).
- [ ] Optionnel : Faire fonctionner ces réseaux dans une application iOS pour avoir des résultats en temps réel.

### Partie II (Attaque des réseaux : *Dodging* puis *Impersonating*)
- [ ] Sur les réseaux collectés (*A* et *B*), essayer de tromper la reconaîssance de manière "naïve" (par exemple une descente de gradient sur quelques pixels d'un panneau STOP jusqu'à ce qu'il ne soit plus reconnu).
- [ ] Étudier l'état de l'art dans ce domaine : réseaux adversaires (***AN***) et attaques *black-box*.
- [ ] Mettre en place ces attaques sur les trois réseaux.
- [ ] Étudier et comparer les résultats.

### Partie III (Renforcement des réseaux)
- [ ] Inventer et mettre en place un renforcement "naïf" contre de telles attaques.
- [ ] Étudier le fonctionnement des réseaux adversaires génératifs (***GANs***) comme technique de renforcement possible.
- [ ] Collecter et expérimenter d'autres stratégies de renforcement existantes.
- [ ] Comparer le succès des attaques sur les trois réseaux, avec diverses techniques de renforcement mises en place. 
- [ ] Conclure

---

## À voir aussi

- Réseaux **RBF** que Ian Goodfellow décrit dans [**6**] comme résistants aux attaques adversaires.

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

> Papier où est essayée l'attaque de panneaux STOP.

- [**8**] **Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition** (2016) [[pdf]](https://www.cs.cmu.edu/~sbhagava/papers/face-rec-ccs16.pdf)

> Comment tromper une reconnaissance faciale avec de simples lunettes ?

### Sources complémentaires

-[**_**] **Traffic Sign Classification Using Deep Inception Based Convolutional Networks** (2015) [[pdf]](https://arxiv.org/pdf/1511.02992.pdf)

> In this work, we propose a novel deep network for traffic sign classification that achieves outstanding performance on GTSRB surpassing all previous methods. Our deep network consists of spatial transformer layers and a modified version of inception module specifically designed for capturing local and global features together. This features adoption allows our network to classify precisely intraclass samples even under deformations. Use of spatial transformer layer makes this network more robust to deformations such as translation, rotation, scaling of input images. Unlike existing approaches that are developed with hand-crafted features, multiple deep networks with huge parameters and data augmentations, our method addresses the concern of exploding parameters and augmentations. We have achieved the state-of-the-art performance of 99.81% on GTSRB dataset.

- [**_**] **Dropout: A Simple Way to Prevent Neural Networks from
Overfitting** (2014), G. Hinton, A. Krizhevsky et al. [[pdf]](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

- [**_**] **Practical Black-Box Attacks against Machine Learning** (2017), I. Goodfellow et al. [[pdf]](https://arxiv.org/pdf/1602.02697v4.pdf)

- [**_**] **NIPS 2016 Tutorial: Generative Adversarial Networks** [[pdf]](https://arxiv.org/pdf/1701.00160v4.pdf)


## Liens utiles

- [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb)
- [Arxiv Sanity Preserver (Recherche de publications)](http://www.arxiv-sanity.com)
- [Glossaire complet et détaillé](http://www.wildml.com/deep-learning-glossary/)

## Dépendances

- Python 3.6
- modules : numpy, tensorflow, keras, scikit-image, pandas, glob