# Contre-attaques adversaires

## Introduction

## Les exemples adversaires

Les réseaux de neurones sont notoirement vulnérables aux attaques par *exemples adversaires* : il s'agit d'entrées inperceptiblement perturbées pour induire en erreur un réseau classificateur.

Plus concrètement, en considérant <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/d6e11b752ad953712a64bc8c88ba6b6a.svg?invert_in_darkmode" align=middle width=36.919905pt height=22.831379999999992pt/> la fonction qui à une image associe la prédiction du réseau, et en considérant une image <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/179d2870187b498a3d368676c476d2c3.svg?invert_in_darkmode" align=middle width=28.52685pt height=21.683310000000006pt/> de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/5bee02fc7a8ca5372bb2192f7bb6a799.svg?invert_in_darkmode" align=middle width=41.002829999999996pt height=24.65759999999998pt/>, on cherche une perturbation <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode" align=middle width=7.873024500000003pt height=14.155350000000013pt/> de norme minimale telle que :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/174ec305bf2bff7a03422b142ec0864e.svg?invert_in_darkmode" align=middle width=219.58694999999997pt height=49.31553pt/></p>

Une méthode d'attaque possible est la suivante. Introduisons <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> la fonction qui à un couple <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/48f63ac3caedbe4ad874af62e676cbfb.svg?invert_in_darkmode" align=middle width=124.28921999999999pt height=24.65759999999998pt/> associe la probabilité (selon le réseau) que l'image appartienne à la catégorie donnée, et considérons une image <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/179d2870187b498a3d368676c476d2c3.svg?invert_in_darkmode" align=middle width=28.52685pt height=21.683310000000006pt/> de catégorie <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode" align=middle width=7.113876000000004pt height=14.155350000000013pt/>. On cherche alors à minimiser par descente de gradient la fonction <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/412737f1db96536b7345d3593320d88d.svg?invert_in_darkmode" align=middle width=41.118825pt height=22.46574pt/> suivante :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/2ae86d9b14696f8b2ed029353701bd85.svg?invert_in_darkmode" align=middle width=426.95399999999995pt height=49.31553pt/></p>

Cette première fonction est expérimentalement peu satisfaisante : l'attaque échoue souvent. Pour pallier celà, on "oblige" la perturbation à grossir avec un quatrième cas de figure, quand <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/563e25d8bdf05f1f9f015f13c49a8b21.svg?invert_in_darkmode" align=middle width=168.77470499999998pt height=24.65759999999998pt/>.

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/86d284c30fcfb843f782bc825b6cf838.svg?invert_in_darkmode" align=middle width=436.5438pt height=69.041775pt/></p>

Cette deuxième fonction produit toujours un exemple adversaire pour un nombre d'étapes de descente de gradient suffisamment élevé (généralement 200 étapes suffisent).

Les fonctions <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/d6e11b752ad953712a64bc8c88ba6b6a.svg?invert_in_darkmode" align=middle width=36.919905pt height=22.831379999999992pt/> (en rouge) et <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> (en bleu) évoluent alors de la manière suivante, en fonction du nombre d'étapes de descente de gradient effectuées :

![Attaque adversaire 1](images/attaque_1@2x.png)

Qualitativement, la norme de la perturbation augmente jusqu'à ce que <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> passe en dessous de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/9b527e843dd83a6485c635f8c4366f78.svg?invert_in_darkmode" align=middle width=29.223975pt height=21.18732pt/>, à partir de quoi la norme diminue en gardant une valeur de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> stabilisée autour de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/358d4d0949e47523757b4bc797ab597e.svg?invert_in_darkmode" align=middle width=21.004665000000006pt height=21.18732pt/>, ce qui s'explique par le choix de cette valeur comme seuil dans la fonction d'erreur.

Cette image peut être qualifiée de "difficile à attaquer" : il a été nécessaire d'augmenter très fortement la norme de la perturbation pour casser la prédiction du réseau, et la norme finale de la perturbation est élevée.

Par comparaison, l'image suivante peut être qualifiée de "facile à attaquer" : bien moins d'étapes ont été nécessaires pour casser la prédiction du réseau, et la norme finale est très basse.

![Attaque adversaire 2](images/attaque_2@2x.png)

## La résistance à une attaque

Pour chaque image, il est possible de quantifier la *résistance* du réseau : il s'agit de la norme minimale d'une perturbation mettant en échec le réseau :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/b9a76c472a8e734d0bb1a242b4f8b356.svg?invert_in_darkmode" align=middle width=393.82694999999995pt height=16.438356pt/></p>

Expérimentalement, les perturbations obtenues par la méthode précédente approche la valeur de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/b482c509424d573ea9b38bd47c807d5f.svg?invert_in_darkmode" align=middle width=27.96816pt height=22.46574pt/> de manière satisfaisante, pour un nombre d'étapes suffisamment grand (autour de 500).

Ume image "facile à attaquer" aura donc une résistance faible, et inversement.

## La résistance comme indicateur de sûreté ?

Considérons un réseau de type <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/61feaa69f87be5f87163b0ef43b33a43.svg?invert_in_darkmode" align=middle width=60.41046pt height=20.09139000000001pt/> (CNN avec Dropout) appliqué au problème de la classification ds chiffres manuscrits de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/328aebe52d5ae49b0a885beb672532f3.svg?invert_in_darkmode" align=middle width=43.15047pt height=20.09139000000001pt/>.

On constate expérimentalement que les images correctement classifiées par le réseau sont "difficiles" à attaquer : On a généralement <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/78ffd9ac5bab82f23541f045c6b946cc.svg?invert_in_darkmode" align=middle width=70.890435pt height=22.46574pt/>. Avec 500 étapes, sur les 250 premières images de validation de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/328aebe52d5ae49b0a885beb672532f3.svg?invert_in_darkmode" align=middle width=43.15047pt height=20.09139000000001pt/>, on obtient la répartition suivante :

![Histogramme 1](images/hist_1@2x.png)

À l'inverse, les images sur lesquelles le réseau se trompe sont faciles à attaquer, avec le plus souvent <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/12548dab16a5b5045413dd5d439700e8.svg?invert_in_darkmode" align=middle width=70.890435pt height=22.46574pt/>. Avec encore 500 étapes, sur les 20 premières images incorrectement classifiées par le réseau, on obtient la répartition suivante :

![Histogramme 2](images/hist_2@2x.png)

On peut donc conjecturer que la résistance est corrélée à la justesse de la classification : une classification correcte correspond à une résistance élevée, et inversement.

## Les contre-attaques adversaires

On observe un autre phénomène : si une attaque adversaire cherche à tromper le réseau, une attaque adversaire sur une image incorrectement classifiée va, le plus souvent, produire une image qui sera alors correctement classifiée ! Ce phénomène se produit en moyenne 80% du temps. On alors parle de *contre-exemple adversaire*.

Une contre attaque adversaire est donc une attaque adversaire sur une image incorrectement classifiée.

## Les contres-attaques adversaires comme méthode pour réduire l'erreur commise

Exploitons les deux phénomènes précédents pour tenter de reduire l'erreur commise par le réseau : Une attaque adversaire est tentée sur chaque image du réseau. Si la résistance est supérieure à un certain critère, on considèrera que la prédiction du réseau est correcte, et sinon on considèrera que le réseau prédit la nouvelle catégorie obtenue.

Avec les 270 images précédentes (250 justes, 20 erreurs), on obtient en fonction du critère choisi :

![Critère](images/critere@2x.png)

On passe ainsi de 20 erreurs à 8 erreurs avec un critère à <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/c549680e9da940f9c4ff8dd9683ada60.svg?invert_in_darkmode" align=middle width=29.223975pt height=21.18732pt/> !

## Un affinement de cette méthode
