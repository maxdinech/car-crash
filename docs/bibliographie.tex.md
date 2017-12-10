# Bibliographie

## 1. État de l'art

M. Nielsen. **Neural Networks and Deep Learning** (2015)
[[web]](http://neuralnetworksanddeeplearning.com)

> Une bonne introduction au Deep Learning, pour débutants. La plupart des concepts présentés sont expliqués en profondeur et démontrés clairement.

I. Goodfellow, Y. Bengio & A. Courville. **Deep Learning** (2016)

> Un livre qui parcourt tous les domaines les plus importants du Deep Learning, une très grosse référence. Il est écrit par trois des plus grands spécialistes en la matière. Très (trop ?) poussé, sert à se renseigner sur un point précis, pas fait pour apprendre.

Y. Le Cun, Y. Bengio & G. Hinton. **Deep Learning**, *Nature* (2015) 
[[pdf]](http://pages.cs.wisc.edu/~dyer/cs540/handouts/deep-learning-nature2015.pdf)

> Un état de l'art du domaine, par trois des plus grands spécialistes, dans la revue Nature.

## 2. Séries de publications sur des thèmes précis

### 2.1 Dropout

G. Hinton, N. Srivastava, A. Krizhevsky & al. **Improving neural networks by preventing co-adaptation of feature detectors** (2012) 
[[arXiv]](https://arxiv.org/abs/1207.0580)

> Première introduction du concept de *dropout*, une manière de réduire l'*overfitting* (sur-adaptation) d'un réseau. Cette technique est utilisée pour la première fois, par les mêmes auteurs, pour concevoir le réseau *AlexNet* (voir plus bas).

N. Srivastava, G. Hinton, A. Krizhevsky & al. **Dropout: A Simple Way to Prevent Neural Networks from Overfitting** (2014)
[[pdf]](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

> Cette (longue) publication démontre plus formellement l'efficacité du dropout.

### 2.2 Batch Normalization

Sergey Ioffe & Christian Szegedy. **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift** (2015) 
[[arXiv]](https://arxiv.org/abs/1502.03167)

> Une technique visant à réduire fortement le temps d’apprentissage d'un réseau.

J. Ba, Jamie R. Kiros & G. Hinton. **Layer Normalization** (2016) 
[[arXiv]](https://arxiv.org/abs/1607.06450)

> Une mise à jour de la technique précédente.

### 2.3 Optimiseurs

I. Sutskever, J. Martens, G. Dahl & G. Hinton. **On the importance of initialization and momentum in deep learning** (2013)
[[pdf]](http://proceedings.mlr.press/v28/sutskever13.pdf)

> Introduction du momentum, une technique d'amélioration de SGD ( descente stochastique de gradient).

D. Kingma, J. Ba. **Adam: A method for stochastic optimization** (2015)

> La technique d'entraînement la plus utilisée aujourd'hui, *Adam*.

### 2.4 Classification d'images (concours ImageNet)

#### AlexNet (Top 5 : 15.3%)

A. Krizhevsky, I. Sutskever and G. Hinton. **ImageNet classification with Deep Convolutional Neural Networks** (2012)
[[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

> Cette publication explique comment un *CNN*, l'architecture *AlexNet*, a pulvérisé le concours annuel de classification d'images ISLVRC (ou ImageNet) en 2012. Depuis, les *CNNs* sont les leaders dans le domaine de la vision par ordinateur.

> Les *CNNs* étaient déjà utilisés alors, mais peu performants. *AlexNet* s'est démarqué par quelques apport notables, détaillés dans la publication, et qui sont devenus des standards par la suite. Notamment :

> - Utilisation de la fonction de transfert ReLU, qui permet un entraînement bien plus rapide du réseau (Section 3.1),
> - L'augmentation de données (Section 4.1),
> - Utilisation du Dropout comme moyen de réduire l'overfitting (Section 4.2)

#### ZF Net (Top 5 : )

M.D. Zeiler and R. Fergus. **Visualizing and Understanding Convolutional Networks** (2013)
[arXiv](https://arxiv.org/abs/1311.2901)

> Le réseau qui a gagné le concours l'année suivante, *ZF Net*.
> Ce réseau n'est pas beaucoup plus innovant que *AlexNet*, mais cette publication est particulièrement intéressante : elle propose des méthodes pour comprendre l'intuition derrière le fonctinnement des *CNNs*, et des techniques de visualisation des *feature maps* produites par les convolutions.

#### VGGNet (Top 5 : )

K. Simonyan and A. Zisserman. **Very Deep Convolutional Networks for Large-Scale Image Recognition** (2014)
[arXiv](https://arxiv.org/abs/1409.1556)

> Un réseau encore plus performant que le précédent : *VGGNet*.
> Les auteurs expliquent comment ils ont gagné en performance: En faisant plus simple et plus profond. Au lieu des convolutions 11x11 de *AlexNet* ou 7x7 de *ZF Net*, ce réseau est entièrement constitué de convolutions 3x3 en cascades (avec des *MaxPool* 2x2 intercalés).

#### GoogleNet (Top 5 : 6.7%)

C. Szegedy et al. **Going Deeper with Convolutions** (2014)
[arXiv](https://arxiv.org/abs/1409.4842)

> Ce papier introduit une nouvelle architecture de réseaux, *Inception*, dont l'un des représentants, *GoogleNet*, profond de 22 couches, gagnant du concours ImageNet en 2014.

> Ce réseau va à l'encontre de tout ce qui était fait jusqu'alors : Au lieu d'empiler un maximum de convolutions et poolings, l'architecture proposée est beaucoup plus compiquée, s'organisant en modules *Inception* mis bout à bout.

#### ResNet (Top 5 : 3.6%)

K. He et al. **Deep Residual Learning for Image Recognition** (2105) 
[arXiv](https://arxiv.org/abs/1512.03385)

> Cette publication introduit le réseau *ResNet*, proposé par Microsoft, gagnant du concours ImageNet 2015.

### 2.5 Exemples adversaires (*Adversarial examples*)

> L'exploration du domaine des exemples adversaires commence la publication suivante :

C. Szegedy, I. Goodfellow & al. **Intriguing Properties of Neural Networks** (2014)
[[arXiv]](https://arxiv.org/abs/1312.6199)

> Les auteurs relèvent deux propriétés "contre-intuitives" des réseaux de neurones, liées à leur modèle d'aprentissage. La deuxième, qui nous intéresse ici, est que les associations entrées-sorties apprises par les réseaux sont fortement discontinues au niveau de l'espace des données.

> Pour mettre en évidence ce phénomène, ils modifient de manière imperceptible une image, et obtiennent une classification erronée avec une assurance élevée.

> Enfin, ils observent un autre phénomène : une même perturbation peut induire en erreur deux réseaux différents mais entraînés sur les mêmes images.

**Explaining and Harnessing Adversarial Examples** (2015), I. Goodfellow & al. [[pdf]](https://arxiv.org/pdf/1412.6572.pdf)

**Practical Black-Box Attacks against Machine Learning** (2017), I. Goodfellow & al. [[pdf]](https://arxiv.org/pdf/1602.02697v4.pdf)

## 3. Panneaux routiers

**Traffic Sign Classification Using Deep Inception Based Convolutional Networks** (2015)
[[arXiv]](https://arxiv.org/abs/1511.02992)

> Classification des panneaux routiers.

**Robust Physical-World Attacks on Machine Learning Models** (2017) [[arXiv]](https://arxiv.org/abs/1707.08945)

> Attaque de panneaux STOP.