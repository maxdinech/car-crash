# Car Crash


*L’efficacité exceptionnelle des réseaux de neurones les rend intéressant dans de nombreux domaines, en particulier celui de la conduite autonome. Mais est-ce bien raisonnable de leur faire autant confiance ?*


## Plan

**Partie I.** Reconnaissance des panneaux de signalisation

**Partie II.** Tromper le réseau : comment hacker un panneau STOP ?

**Partie III.** Des pistes pour prémunir contre de telles attaques


## Roadmap

### Partie I (Mise en place des réseaux)
> 1. Comprendre le fonctionnement d'un réseau de neurones basique (***MLP***) ([**1**], Ch. 1 et [**2**]).
> 2. Mettre en place un MLP de classification des chiffres manuscrits de MNIST, sans framework, en Python. ([**1**], Ch. 1)
> 3. Comprendre le fonctionnement des réseau neuronaux convolutifs (***CNN***).
> 4. Mettre en place un CNN avec **TensorFlow** pour la reconaîssance de MNIST
> 5. Pareil, pour la classification des panneaux (GTSRB) (Réseau A).
> 7. Trouver (ailleurs) un réseau "sûr" de classification des panneaux de GTSRB qui servira de réference (Réseau B).
> 8. Faire fonctionner ces réseaux dans une application iOS pour avoir des résultats en temps réel.

### Partie II (Attaque des réseaux)
> 9. Sur les réseaux A et B, essayer une descente de gradient sur les pixels d'un panneau STOP reconnu par les réseaux jusqu'à les tromper. Étude mathématique de l'efficacité de cette descente.
> 10. Étudier le fonctionnement des réseaux adversaires (***AN***) et adversaires génératifs (***GANs***).
> 11. Essayer de mettre en place une attaque par AN sur les réseaux A et B.
> 13. Étudier et comparer les résultats.

### Partie III (Renforcement des réseaux)
> 13. Étudier les renforcements possibles des réseaux A et B, par exemple par des GANs.
> 14. ? Mettre en place ces stratégies, et nouvelle tentative de "hack"
> 15. Conclure

### Bonus ?
> 16. Pourquoi pas commencer par la détection du panneau dans le champ de vision pour ensuite l'identifier (sur la version iOS) ? Voir GTS**D**B.
> 17. Contacter Renault-Nissan pour voir leurs modèles de reconaîssance de panneaux ?


## Bibliographie

### Livres

- [**1**] **Neural Networks and Deep Learning** (2015), Michael A. Nielsen [[web]](http://neuralnetworksanddeeplearning.com)

> Très bonne introduction au DL, avec des applications directes en Python.

- [**2**] **Deep Learning** (2016), Ian Goodfellow, Yoshua Bengio and Aaron Courville [[web]](http://deeplearningbook.org)

> Livre exausthif sur le Deep Learning, une très grosse référence.

### Publications

- [**_**] **ImageNet Classification with Deep Convolutional Neural Networks** (2012), A. Krizhevsky et al. [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [**_**] **Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition** (2016) [[pdf]](https://www.cs.cmu.edu/~sbhagava/papers/face-rec-ccs16.pdf)
- [**_**] **Robust Physical-World Attacks on Machine Learning Models** (2017) [[pdf]](https://arxiv.org/pdf/1707.08945.pdf)
- [**_**] **Practical Black-Box Attacks against Machine Learning** [[pdf]](https://arxiv.org/pdf/1602.02697v4.pdf)
- [**_**] **Explaining and Harnessing Adversarial Examples** [[pdf]](https://arxiv.org/pdf/1412.6572.pdf)
- [**_**] **NIPS 2016 Tutorial: Generative Adversarial Networks** [[pdf]](https://arxiv.org/pdf/1701.00160v4.pdf)


## Liens utiles

- [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb)
- [Arxiv Sanity Preserver (Recherche de publications)](http://www.arxiv-sanity.com)
- [Très bon glossaire](http://www.wildml.com/deep-learning-glossary/)

