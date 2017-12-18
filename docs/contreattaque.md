# La résistance aux attaques adversaires comme indicateur de sûreté

## Introduction

On introduit la notion de *résistance*, qui quantifie la difficilté à tromper un réseau de neurones classificateur avec un exemple adversaire créé à partir d'une image donnée. On cherchera d'abord plusieurs expressions possibles de la résistance, et on essaiera d'utiliser ce concept comme méthode pour détecter les exemples adversaires et améliorer la performance d'un réseau classificateur.

## 1. Les attaques adversaires

### 1.1 Les exemples adversaires

Les réseaux de neurones sont notoirement vulnérables aux attaques par *exemples adversaires* [1, 2] : il s'agit d'entrées inperceptiblement perturbées pour induire en erreur un réseau classificateur.

Plus concrètement, en considérant <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/d6e11b752ad953712a64bc8c88ba6b6a.svg?invert_in_darkmode" align=middle width=36.919905pt height=22.831379999999992pt/> la fonction qui à une image associe la prédiction du réseau, et en considérant une image <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/179d2870187b498a3d368676c476d2c3.svg?invert_in_darkmode" align=middle width=28.52685pt height=21.683310000000006pt/> de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/5bee02fc7a8ca5372bb2192f7bb6a799.svg?invert_in_darkmode" align=middle width=41.002829999999996pt height=24.65759999999998pt/>, on cherche une perturbation <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode" align=middle width=7.873024500000003pt height=14.155350000000013pt/> de norme minimale telle que :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/174ec305bf2bff7a03422b142ec0864e.svg?invert_in_darkmode" align=middle width=219.58694999999997pt height=49.31553pt/></p>

### 1.2 Les attaques adversaires

On cherche un algorithme qui détermine un exemple adversaire à partir d'une image donnée. On dit qu'un tel algorithme réalise une *attaque adversaire*.

Une méthode d'attaque possible est la suivante. Introduisons <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> la fonction qui à une image de catégorie <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode" align=middle width=7.113876000000004pt height=14.155350000000013pt/> associe la probabilité (selon le réseau) que l'image appartienne cette catégorie ; et soit une image <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/179d2870187b498a3d368676c476d2c3.svg?invert_in_darkmode" align=middle width=28.52685pt height=21.683310000000006pt/> de catégorie <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode" align=middle width=7.113876000000004pt height=14.155350000000013pt/>. On cherche alors à minimiser par descente de gradient (avec un taux d'apprentissage <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/fb9bf6a9b8dad77d5d93ff9b1ed76a45.svg?invert_in_darkmode" align=middle width=63.934695pt height=26.76201000000001pt/>) la fonction <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/412737f1db96536b7345d3593320d88d.svg?invert_in_darkmode" align=middle width=41.118825pt height=22.46574pt/> suivante :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/2ae86d9b14696f8b2ed029353701bd85.svg?invert_in_darkmode" align=middle width=426.95399999999995pt height=49.31553pt/></p>

Cette première fonction est expérimentalement peu satisfaisante, car l'attaque échoue souvent. La perturbation reste "bloquée" en <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode" align=middle width=8.219277000000005pt height=21.18732pt/>, et n'évolue pas. Pour pallier celà, on oblige la perturbation à grossir en ajoutant un troisième cas de figure, quand <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/563e25d8bdf05f1f9f015f13c49a8b21.svg?invert_in_darkmode" align=middle width=168.77470499999998pt height=24.65759999999998pt/>, c'est à dire quand la perturbation n'est pas du tout satisfaisante :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/86d284c30fcfb843f782bc825b6cf838.svg?invert_in_darkmode" align=middle width=436.5438pt height=69.041775pt/></p>

Cette deuxième fonction produit toujours un exemple adversaire pour un nombre d'étapes de descente de gradient suffisamment élevé (généralement 200 étapes suffisent).

### 1.3 Quelques résultats

On réalise l'attaque adversaire en effectuant <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/25df05ed4b8476cb9a1a3db76ae8f22c.svg?invert_in_darkmode" align=middle width=24.657765pt height=21.18732pt/> étapes de descente du gradient de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/3a5c9e3056428b808beee92e988ff843.svg?invert_in_darkmode" align=middle width=41.118825pt height=22.46574pt/>, avec un taux d'apprentissage <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/fb9bf6a9b8dad77d5d93ff9b1ed76a45.svg?invert_in_darkmode" align=middle width=63.934695pt height=26.76201000000001pt/>.

Les fonctions <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/d6e11b752ad953712a64bc8c88ba6b6a.svg?invert_in_darkmode" align=middle width=36.919905pt height=22.831379999999992pt/> (en rouge) et <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> (en bleu) évoluent alors de la manière suivante, en fonction du nombre d'étapes de descente de gradient effectuées :

![Attaque adversaire 1](images/attaque_1.png)

Qualitativement, la norme de la perturbation augmente jusqu'à ce que <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> passe en dessous de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/9b527e843dd83a6485c635f8c4366f78.svg?invert_in_darkmode" align=middle width=29.223975pt height=21.18732pt/>, à partir de quoi la norme diminue en gardant une valeur de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> stabilisée autour de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/358d4d0949e47523757b4bc797ab597e.svg?invert_in_darkmode" align=middle width=21.004665000000006pt height=21.18732pt/>.

Cette image peut être qualifiée de "difficile à attaquer" : il a été nécessaire d'augmenter très fortement la norme de la perturbation pour réussir à casser la prédiction du réseau ; et la norme finale de la perturbation est élevée.

L'image suivante, au contraire, peut être qualifiée de "facile à attaquer" : bien moins d'étapes ont été nécessaires pour casser la prédiction du réseau, et la norme finale est très basse.

![Attaque adversaire 2](images/attaque_2.png)

On voit nettement ici l'influence de la valeur du seuil à <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/358d4d0949e47523757b4bc797ab597e.svg?invert_in_darkmode" align=middle width=21.004665000000006pt height=21.18732pt/> dans la fonction <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/f8d449532805c3c7a4839aa3da6af335.svg?invert_in_darkmode" align=middle width=34.566345pt height=22.46574pt/>. Dès que <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> est en dessous de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/358d4d0949e47523757b4bc797ab597e.svg?invert_in_darkmode" align=middle width=21.004665000000006pt height=21.18732pt/>, l'alogorithme a pour seul objectif de réduire la norme de la perturbation, et fatalement <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> repasse au dessus de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/358d4d0949e47523757b4bc797ab597e.svg?invert_in_darkmode" align=middle width=21.004665000000006pt height=21.18732pt/>. Il s'agit alors de réduire à la fois <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/b9a12c4d22f51e9aef9d97ab1b2351a2.svg?invert_in_darkmode" align=middle width=24.31143pt height=24.65759999999998pt/> et <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/>, jusqu'à ce que <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> repasse en dessous de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/358d4d0949e47523757b4bc797ab597e.svg?invert_in_darkmode" align=middle width=21.004665000000006pt height=21.18732pt/>...

### 1.4 Un peu plus de résultats

D'abord 5 images "difficiles" à attaquer :

![Attaques adversaire 1](images/attaques_1.png)

Puis 5 images "faciles" à attaquer.

![Attaques adversaire 2](images/attaques_2.png)

Ces images n'ont pas été choisies au hasard : celles du haut sont les 5 permières de la base de donnée (sans erreurs), et celles du bas correspondent aux 5 premières erreurs de classification commises par le réseau.

Dans ces deux cas de figure, l'évolution de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/b9a12c4d22f51e9aef9d97ab1b2351a2.svg?invert_in_darkmode" align=middle width=24.31143pt height=24.65759999999998pt/> est drastiquement différente. L'objectif de ce travail sera de mettre en évidence une possible corrélation entre la difficulté à attaquer une image (pic haut, norme finale élevée) et la justesse de la prédiction du réseau. On introduit pour cela le concept de *résistance*.

## 2. La résistance <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/b482c509424d573ea9b38bd47c807d5f.svg?invert_in_darkmode" align=middle width=27.96816pt height=22.46574pt/>

### 2.1 La résistance à une attaque

Pour chaque image, on essaie de quantifier la *résistance* du réseau. Plusieurs définitions sont possibles, par exemple la norme de la perturbation minimale mettant en échec le réseau :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/e40aa98f986b784f6ceb2386a9adc269.svg?invert_in_darkmode" align=middle width=398.62185pt height=16.438356pt/></p>

On peut également utiliser comme valeur de la résistance la norme finale obtenue après un certain nombre d'étapes dans l'attaque adversaire précédente :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/fe81a680b03663a778fabe8e2d2167df.svg?invert_in_darkmode" align=middle width=458.0202pt height=16.438356pt/></p>

Ou bien la hauteur du pic de la norme de la perturbation :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/519c070f1d2e6984d0722c98cb57d7ba.svg?invert_in_darkmode" align=middle width=251.06895pt height=16.438356pt/></p>

Ou encore le nombre d'étapes qu'il a fallu pour abaisser <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> à <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/358d4d0949e47523757b4bc797ab597e.svg?invert_in_darkmode" align=middle width=21.004665000000006pt height=21.18732pt/>:

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/35533f67ba691d5579a4b2d54d6cf30b.svg?invert_in_darkmode" align=middle width=416.46165pt height=16.438356pt/></p>

### 2.2 Une corrélation avec la fiabilité de la prédiction ?

Considérons un réseau de type <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/61feaa69f87be5f87163b0ef43b33a43.svg?invert_in_darkmode" align=middle width=60.41046pt height=20.09139000000001pt/> (CNN avec Dropout) appliqué au problème de la classification ds chiffres manuscrits de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/328aebe52d5ae49b0a885beb672532f3.svg?invert_in_darkmode" align=middle width=43.15047pt height=20.09139000000001pt/>, et utilisons les deuxième et troisième définition de la résistance : <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/aad0bbd7789ee69fbedd1a6506359296.svg?invert_in_darkmode" align=middle width=69.36336pt height=22.46574pt/> et <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/72328aeee5790e07313860e0c6f58ed8.svg?invert_in_darkmode" align=middle width=54.21768pt height=22.46574pt/>.

Observons la répartition des résistances, d'abord sur 200 images correctement classifiées (**V**), puis sur les 84 incorrectement classifiées (**F**):

| Plage                    | V - <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/107c8992400f43c183ed2eeb19e4217f.svg?invert_in_darkmode" align=middle width=39.614354999999996pt height=22.46574pt/> | F - <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/107c8992400f43c183ed2eeb19e4217f.svg?invert_in_darkmode" align=middle width=39.614354999999996pt height=22.46574pt/> | V - <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/72328aeee5790e07313860e0c6f58ed8.svg?invert_in_darkmode" align=middle width=54.21768pt height=22.46574pt/> | F - <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/72328aeee5790e07313860e0c6f58ed8.svg?invert_in_darkmode" align=middle width=54.21768pt height=22.46574pt/> |
| :----------------------- | :---------: | :---------: | :-------------: | :-------------: |
| <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/fb7eb5acabd354e302742e8aa3a9cec8.svg?invert_in_darkmode" align=middle width=54.794520000000006pt height=24.65759999999998pt/>      |     0%      |   **41%**   |       0%        |       6%        |
| <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/3d1f0737accbc0f1a99b72397a391aef.svg?invert_in_darkmode" align=middle width=67.579875pt height=24.65759999999998pt/>      |     3%      |   **48%**   |       1%        |     **68%**     |
| <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/daaa8348495693d7680a93730e9e9a29.svg?invert_in_darkmode" align=middle width=54.794520000000006pt height=24.65759999999998pt/>        |   **13%**   |     9%      |       3%        |     **19%**     |
| <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/d9796d77c04c15688a33cbfd4b5489a4.svg?invert_in_darkmode" align=middle width=42.009pt height=24.65759999999998pt/>        |   **46%**   |     2%      |     **12%**     |       5%        |
| <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/352280840cdf2a21f3b889b89af852ed.svg?invert_in_darkmode" align=middle width=42.009pt height=24.65759999999998pt/>        |   **29%**   |     0%      |     **53%**     |       2%        |
| <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/e5a5873047aaf5172d484522d6aaeee4.svg?invert_in_darkmode" align=middle width=63.013664999999996pt height=24.65759999999998pt/> |     9%      |     0%      |     **31%**     |       0%        |

Effectuons la même étude sur la fonction <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/81f5383edfff1a9f81de4a1017d20075.svg?invert_in_darkmode" align=middle width=52.409940000000006pt height=22.46574pt/> :

| Plage                    | V - <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/81f5383edfff1a9f81de4a1017d20075.svg?invert_in_darkmode" align=middle width=52.409940000000006pt height=22.46574pt/> | F - <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/81f5383edfff1a9f81de4a1017d20075.svg?invert_in_darkmode" align=middle width=52.409940000000006pt height=22.46574pt/> |
| :----------------------- | :-------------: | :-------------: |
| <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/747a4cbec1c8d2549ebeebee9c337947.svg?invert_in_darkmode" align=middle width=50.228145000000005pt height=24.65759999999998pt/>       |       12%       |                 |
| <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/82c737c0a54ee74a4fd4d3f7bed6df80.svg?invert_in_darkmode" align=middle width=58.447455000000005pt height=24.65759999999998pt/>       |       77%       |                 |
| <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/9e9e932c3e9aad939ad4316e8c58d4ab.svg?invert_in_darkmode" align=middle width=66.6666pt height=24.65759999999998pt/>      |       10%       |                 |
| <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/7980a06fc80e5fbaf09b201aae15aaf8.svg?invert_in_darkmode" align=middle width=74.88591000000001pt height=24.65759999999998pt/>      |       1%        |                 |
| <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/136b872f645c5ab0bf6ab438538d1c54.svg?invert_in_darkmode" align=middle width=79.45212pt height=24.65759999999998pt/> |       0%        |                 |

Une corrélation se dessine : les images correctement classifiées sont en général de résistance plus élecée que les images sur lesquelles le réseau se trompe.

## 3. Les contre-attaques adversaires

On observe un autre phénomène : si une attaque adversaire cherche à tromper le réseau, une attaque adversaire sur une image incorrectement classifiée va, le plus souvent, produire une image qui sera correctement classifiée ! On parlera alors de *contre-exemple adversaire*.

Une contre attaque adversaire est donc une attaque adversaire sur une image incorrectement classifiée, dans l'espoir que la nouvelle catégorie soit la vraie.

Toujours avec le même réseau, observons les catégories obtenues par les contre attaques adversaires.

Sur les <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/0f5e966d662e2f0174b478b93259c578.svg?invert_in_darkmode" align=middle width=16.438455000000005pt height=21.18732pt/> erreurs commises, <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/1cab53fc19941e038c4057835191ee7c.svg?invert_in_darkmode" align=middle width=16.438455000000005pt height=21.18732pt/> des contre attaques adversaires donnent la bonne catégorie, soit dans <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/2ec6c7a0f9c6a213c74a8b8f5e907fba.svg?invert_in_darkmode" align=middle width=30.137085000000006pt height=24.65759999999998pt/> des cas !

## 4. Une méthode pour réduire l'erreur du réseau ?

Exploitons les deux phénomènes précédents pour tenter de reduire l'erreur commise par le réseau : Une attaque adversaire est tentée sur chaque image du réseau. Si la résistance est supérieure à un certain critère, on considèrera que la prédiction du réseau est correcte, et sinon on considèrera que le réseau prédit la nouvelle catégorie obtenue.

Avec les 270 images précédentes (250 justes, 20 erreurs), on obtient en fonction du critère choisi :

![Critère](images/critere.png)

On passe ainsi de 20 erreurs à 8 erreurs avec un critère à <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/4ad3f8dbe17211cba65152798494f248.svg?invert_in_darkmode" align=middle width=29.223975pt height=21.18732pt/> !

## 



## Bibliographie

[1] N. Srivastava, G. Hinton, A. Krizhevsky & al. JMLP, **Dropout: A Simple Way to Prevent Neural Networks from Overfitting.** Volume 15 (2014), Pages 1929-1958

[6] A. Krizhevsky, I. Sutskever & G. Hinton. NIPS'12 Proceedings, **ImageNet Classification with Deep Convolutional Neural Networks .** Volume 1 (2012), Pages 1097-1105