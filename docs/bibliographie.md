# État de l'art

## 1. Quelques noms à connaître

- **Geoffrey Hinton:**
- **Yann Le Cun:**
- **Yoshua Bengio:**
- **Ian Goodfellow:**
- **Alex Krievshki:**

## Livres et publications "générales"

- Michael Nielsen. **Neural Networks and Deep Learning** (2015)
[[web]](http://neuralnetworksanddeeplearning.com)

Une bonne introduction au Deep Learning, pour débutants. La plupart des concepts présentés sont expliqués en profondeur et **démontrés** clairement. Chapitres intéressant : démonstration du fonctionnement de la descente de gradient (Ch. 2) et techniques d'ajustement des hyperparamètres (Ch. 4 ?).

- Ian Goodfellow, Yoshua Bengio & Aaron Courville. **Deep Learning** (2016)
[[web]](http://deeplearningbook.org)

Un livre qui se veut un r"sumé des domaines importants du Deep Learning, une très grosse référence. Il est écrit par trois des plus grands spécialistes du domaine. Très (trop ?) poussé, sert à se renseigner sur un point précis, pas fait pour apprendre.

- [**3**] **Deep Learning**, *Nature* (2015), Yann LeCun, Yoshua Bengio & Geoffrey Hinton [[pdf]](http://pages.cs.wisc.edu/~dyer/cs540/handouts/deep-learning-nature2015.pdf)

Une rapide présentation Deep Learning (backprop, MLP, CNN, RNN, LSTM), dans la prestigieuse revue Nature.


# Séries de publications sur des thèmes précis

## Dropout

- G.E. Hinton, N. Srivastava, A. Krizhevsky & al. **Improving neural networks by preventing co-adaptation of feature detectors** (2012) [[arXiv]](https://arxiv.org/abs/1207.0580)

Première introduction du concept de *"dropout"*, défini comme une manière de réduire l'*overfitting*. Cette technique est utilisée pour la première fois, peu après, par les mêmes auteurs pour concevoir *AlexNet*.

- N. Srivastava, G. Hinton, A. Krizhevsky & al. **Dropout: A Simple Way to Prevent Neural Networks from Overfitting** (2014) [[pdf]](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

Cette publication démontre plus formellement la correction et l'efficacité du *dropout*.


## Optimiseurs (à compléter)

- **Adam: A method for stochastic optimization** (2015), D. Kingma, J. Ba

## BatchNorm

> Publications à venir...

## Classification d'images et concours ImageNet

- A. Krizhevsky, I. Sutskever, and G.E. Hinton. **ImageNet classification with Deep Convolutional Neural Networks** (2012)
[[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Cette publication explique comment un CNN, l'architecture *AlexNet*, a pulvérisé le concours annuel de classification d'images *ImageNet* en 2012. Depuis, les CNNs sont les leaders dans ce domaine de la vision par ordinateur.

Beaucoup de choses intéressantes :
	- Première "vraie" utilisation de la technique du Dropout
 	- Première "vraie" utilisation de la fonction de transfert ReLU
 	- Intérêt de l'augmentation de données



## Réseaux adversaires génératifs

- [**6**] **Generative adversarial nets** (2014), I. Goodfellow & al. [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

Une publication d'une importance capitale, qui a lancé l'étude des **GANs** (que Yann Le Cun décrit tout de même comme la plus grosse avancée des 10 dernières années en Deep Learning !).

Cette technique permet d'utiliser les réseaux de neurones comme générateurs d'images, en entraînant en parallèle un réseau *Générateur* et un réseau *Discriminateur*.


### Sources complémentaires

- [**_**] **Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition** (2016) [[pdf]](https://www.cs.cmu.edu/~sbhagava/papers/face-rec-ccs16.pdf)

Comment tromper une reconnaissance faciale avec de simples lunettes ?

-[**_**] **Traffic Sign Classification Using Deep Inception Based Convolutional Networks** (2015) [[pdf]](https://arxiv.org/pdf/1511.02992.pdf)

In this work, we propose a novel deep network for traffic sign classification that achieves outstanding performance on GTSRB surpassing all previous methods. Our deep network consists of spatial transformer layers and a modified version of inception module specifically designed for capturing local and global features together. This features adoption allows our network to classify precisely intraclass samples even under deformations. Use of spatial transformer layer makes this network more robust to deformations such as translation, rotation, scaling of input images. Unlike existing approaches that are developed with hand- crafted features, multiple deep networks with huge parameters and data augmentations, our method addresses the concern of exploding parameters and augmentations. We have achieved the state-of-the-art performance of 99.81% on GTSRB dataset.


## Adversarial examples and adversarial training

L'exploration du domaine des exemples adversaires commence avec cette publication :

- C. Szegedy, I. Goodfellow & al. **Intriguing Properties of Neural Networks** (2014) 

Les auteurs relèvent deux propriétés "contre-intuitives" des réseaux de neurones, liées à leur modèle d'aprentissage. La deuxième, qui nous intéresse ici, est que les associations entrées-sorties apprises par les réseaux sont fortement discontinues au niveau de l'espace des données.

Pour mettre en évidence ce phénomène, ils modifient de manière imperceptible une image, et obtiennent une classification erronée avec une assurance élevée.

Enfin, ils observent un autre phénomène : une même perturbation peut induire en erreur deux réseaux différents mais entraînés sur les mêmes images.






- [**_**] **Practical Black-Box Attacks against Machine Learning** (2017), I. Goodfellow & al. [[pdf]](https://arxiv.org/pdf/1602.02697v4.pdf)


- [**7**] **Explaining and Harnessing Adversarial Examples** (2015), I. Goodfellow & al. [[pdf]](https://arxiv.org/pdf/1412.6572.pdf)

> Cette publication s'intéresse à un phénomène particulier : la faiblesse des réseaux de neurones face aux examples adversaires (*adversarial examples*). Il s'agit d'entrées calculées de sorte à tromper la prédiction d'un réseau classificateur. On les obtient en appliquant des perturbation les plus faibles possibles sur une image initialement correctement reconnue, de sorte à obtenir une réponse fausse d'assurance quasi-certaine.

> Cette étude explique que cette faiblesse est dûe à la nature linéaire des réseaux de neurones, et introduit une méthode simple et efficace de génération d'examples adversaires, qui servent ensuite à renforcer la fiabilité d'un réseau (testée sur MNIST dans cette publication).

- [**8**] **Robust Physical-World Attacks on Machine Learning Models** (2017)
[[pdf]](https://arxiv.org/pdf/1707.08945.pdf)

> Papier où est essayée l'attaque de panneaux STOP.