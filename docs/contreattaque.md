# Contre-attaques adversaires

## Introduction

## Les exemples adversaires
Les réseaux de neurones sont notoirement vulnérables aux attaques par *exemples adversaires* : il s'agit d'entrées inperceptiblement perturbées pour induire en erreur un réseau classificateur.

Plus concrètement, en considérant <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/d6e11b752ad953712a64bc8c88ba6b6a.svg?invert_in_darkmode" align=middle width=36.919905pt height=22.831379999999992pt/> la fonction qui à une image associe la prédiction du réseau, et en considérant une image <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/179d2870187b498a3d368676c476d2c3.svg?invert_in_darkmode" align=middle width=28.52685pt height=21.683310000000006pt/> de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/5bee02fc7a8ca5372bb2192f7bb6a799.svg?invert_in_darkmode" align=middle width=41.002829999999996pt height=24.65759999999998pt/>, on cherche une perturbation <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode" align=middle width=7.873024500000003pt height=14.155350000000013pt/> de norme minimale telle que :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/174ec305bf2bff7a03422b142ec0864e.svg?invert_in_darkmode" align=middle width=219.58694999999997pt height=49.31553pt/></p>

Une méthode d'attaque possible est la suivante. Introduisons <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> la fonction qui à un couple <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/48f63ac3caedbe4ad874af62e676cbfb.svg?invert_in_darkmode" align=middle width=124.28921999999999pt height=24.65759999999998pt/> associe la probabilité (selon le réseau) que l'image appartienne à la catégorie donnée, et considérons une image <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/179d2870187b498a3d368676c476d2c3.svg?invert_in_darkmode" align=middle width=28.52685pt height=21.683310000000006pt/> de catégorie <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode" align=middle width=7.113876000000004pt height=14.155350000000013pt/>. On cherche alors à minimiser par descente de gradient la fonction <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/412737f1db96536b7345d3593320d88d.svg?invert_in_darkmode" align=middle width=41.118825pt height=22.46574pt/> suivante :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/2871ac7e396d5a4c24b491b283edb6a4.svg?invert_in_darkmode" align=middle width=428.3234999999999pt height=69.041775pt/></p>

Cette première fonction est expérimentalement peu satisfaisante : l'attaque échoue souvent. Pour pallier celà, on "oblige" la perturbation à grossir avec un quatrième cas de figure, quand <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/563e25d8bdf05f1f9f015f13c49a8b21.svg?invert_in_darkmode" align=middle width=168.77470499999998pt height=24.65759999999998pt/>.

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/a6dd4c694e07d132e2ae279442b3739f.svg?invert_in_darkmode" align=middle width=436.5438pt height=88.76801999999999pt/></p>

Cette deuxième fonction produit toujours un exemple adversaire pour un nombre d'étapes de descente de gradient suffisamment élevé (généralement 200 étapes suffisent).

Pour chaque image, il est possible de quantifier la *résistance* du réseau : il s'agit de la norme minimale d'une perturbation mettant en échec le réseau :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/b9a76c472a8e734d0bb1a242b4f8b356.svg?invert_in_darkmode" align=middle width=393.82694999999995pt height=16.438356pt/></p>

Expérimentalement, les perturbations obtenues par les méthodes précédentes approchent la valeur de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/b482c509424d573ea9b38bd47c807d5f.svg?invert_in_darkmode" align=middle width=27.96816pt height=22.46574pt/> de manière satisfaisante.

## La résistance comme indicateur de sûreté ?

Considérons un réseau de type <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/61feaa69f87be5f87163b0ef43b33a43.svg?invert_in_darkmode" align=middle width=60.41046pt height=20.09139000000001pt/> (CNN avec Dropout) appliqué au problème de la classification ds chiffres manuscrits de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/328aebe52d5ae49b0a885beb672532f3.svg?invert_in_darkmode" align=middle width=43.15047pt height=20.09139000000001pt/>.

On constate expérimentalement que les images correctement classifiées par le réseau sont "difficiles" à attaquer : On a généralement <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/a3a1d86d0e372d137485db55387dce19.svg?invert_in_darkmode" align=middle width=58.104915pt height=22.46574pt/>. Avec 500 étapes, sur les 250 premières images de validation de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/328aebe52d5ae49b0a885beb672532f3.svg?invert_in_darkmode" align=middle width=43.15047pt height=20.09139000000001pt/>, on obtient la répartition suivante :

![Histogramme 1](images/hist_1.png)

À l'inverse, les images sur lesquelles le réseau se trompe sont faciles à attaquer, avec le plus souvent <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/8465fd591d6c57b78a48087f5fe9e85b.svg?invert_in_darkmode" align=middle width=70.890435pt height=22.46574pt/>. Avec encore 500 étapes, sur les 20 premières images incorrectement classifiées par le réseau, on obtient la répartition suivante :

![Histogramme 1](images/hist_2.png)

Blabla

Avec les 270 images précédentes (150 justes, 20 erreurs), on bitient en fonction du critère choisi :

![Critère](images/critere.png)