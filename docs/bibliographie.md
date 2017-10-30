# Bibliographie commentée

## Livres

- [**1**] **Neural Networks and Deep Learning** (2015), Michael A. Nielsen [[web]](http://neuralnetworksanddeeplearning.com)

> Une bonne introduction au Deep Learning, pour débutants. Tous les concepts
présentés sont expliqués en profondeur et démontrés. Chapitres intéressant :
démonstration du fonctionnement de la descente de gradient (Ch. 2) et techniques
d'ajustement des hyperparamètres.

- [**2**] **Deep Learning** (2016), Ian Goodfellow, Yoshua Bengio & Aaron Courville [[web]](http://deeplearningbook.org)

> Livre exausthif sur le Deep Learning, une très grosse référence (écrit par trois des plus grands spécialistes du domaine, et ensencé par les autres). Très (trop ?) poussé, sert à se renseigner sur un point précis, pas fait pour apprendre.


## Publications

- [**3**] **Deep Learning**, *Nature* (2015), Yann LeCun, Yoshua Bengio & Geoffrey Hinton [[pdf]](http://pages.cs.wisc.edu/~dyer/cs540/handouts/deep-learning-nature2015.pdf)

> Un “état de l'art” du Deep Learning par deux spécialistes du domaine dans la prestigieuse revue *Nature*. Pas de grand intérêt scientifique (pas une publication marquante), mais symbolique.

- [**4**] **ImageNet Classification with Deep Convolutional Neural Networks** (2012), A. Krizhevsky et al. [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

> Cette publication explique comment un CNN, avec une architecture appelée *AlexNet* a pulvérisé le concours annuel de classification d'images ImageNet (22000 catégories !) en 2012. Depuis, les CNNs sont les leaders dans ce domaine de la vision par ordinateur.
> Il y justifie notamment l'intérêt de la fonction *ReLU*, et de l'augmentation des données.

- [**5**] **Generative adversarial nets** (2014), I. Goodfellow et al. [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

> Une publication d'une importance capitale, qui a lancé l'étude des **GANs** (que Yann Le Cun décrit tout de même comme la plus grosse avancée des 10 dermières années en Deep Learning !).
> Cette technique permet d'utiliser les réseaux de neurones comme générateurs d'image, en entraînant en parallèle un réseau *Générateur* et un réseau *Discriminateur*.

- [**6**] **Explaining and Harnessing Adversarial Examples** (2015), I. Goodfellow et al. [[pdf]](https://arxiv.org/pdf/1412.6572.pdf)

> Cette publication s'intéresse à un phénomène particulier : la faiblesse des réseaux de neurones face aux *examples adversaires*, qui sont des entrées qui cherchent à tromper la réponse d'un réseau classificateur. On les obtient en appliquant des perturbation les plus faibles possibles sur une image initialement correctement reconnue, de sorte à obtenir une réponse fausse et d'assurance quasi-certaine.
> Cette étude explique que cette faiblesse est dûe à la nature linéaire des réseaux de neurones, et introduit une méthode simple et efficace de génération d'examples adversaires, qui servent ensuite à renforcer la fiabilité d'un réseau (vérifiée sur MNIST dans cette publication).

- [**7**] **Robust Physical-World Attacks on Machine Learning Models** (2017) [[pdf]](https://arxiv.org/pdf/1707.08945.pdf)

> Papier où est essayée l'attaque de panneaux STOP.

- [**8**] **Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition** (2016) [[pdf]](https://www.cs.cmu.edu/~sbhagava/papers/face-rec-ccs16.pdf)

> Comment tromper une reconnaissance faciale avec de simples lunettes ?

### Sources complémentaires

-[**_**] **Traffic Sign Classification Using Deep Inception Based Convolutional Networks** (2015) [[pdf]](https://arxiv.org/pdf/1511.02992.pdf)

> In this work, we propose a novel deep network for traffic sign classification that achieves outstanding performance on GTSRB surpassing all previous methods. Our deep network consists of spatial transformer layers and a modified version of inception module specifically designed for capturing local and global features together. This features adoption allows our network to classify precisely intraclass samples even under deformations. Use of spatial transformer layer makes this network more robust to deformations such as translation, rotation, scaling of input images. Unlike existing approaches that are developed with hand-crafted features, multiple deep networks with huge parameters and data augmentations, our method addresses the concern of exploding parameters and augmentations. We have achieved the state-of-the-art performance of 99.81% on GTSRB dataset.

- [**_**] **Dropout: A Simple Way to Prevent Neural Networks from Overfitting** (2014), G. Hinton, A. Krizhevsky et al. [[pdf]](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

- [**_**] **Practical Black-Box Attacks against Machine Learning** (2017), I. Goodfellow et al. [[pdf]](https://arxiv.org/pdf/1602.02697v4.pdf)

- [**_**] **NIPS 2016 Tutorial: Generative Adversarial Networks** [[pdf]](https://arxiv.org/pdf/1701.00160v4.pdf)
