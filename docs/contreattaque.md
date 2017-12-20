% Résistance aux attaques et corrections adversaires

## Résumé

On introduit la notion de *résistance*, qui quantifie la difficulté à tromper un réseau de neurones classificateur avec un exemple adversaire créé à partir d'une image donnée.

Ce concept offre trois apports intéressants :

- une méthode de diminution de l'erreur commise par un réseau donné,
- une méthode de détection des exemples adversaires,
- Une quantification de la certitude d'une prédiction.

## 1. Les attaques adversaires

### 1.1 Les exemples adversaires

Les réseaux de neurones sont notoirement vulnérables aux *exemples adversaires* [1] : il s'agit d'entrées imperceptiblement perturbées pour induire en erreur un réseau classificateur.

Plus concrètement, en considérant <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/d6e11b752ad953712a64bc8c88ba6b6a.svg?invert_in_darkmode" align=middle width=36.919905pt height=22.831379999999992pt/> la fonction qui à une image associe la catégorie prédite par réseau ; et en considérant une image <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/179d2870187b498a3d368676c476d2c3.svg?invert_in_darkmode" align=middle width=28.52685pt height=21.683310000000006pt/> de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/5bee02fc7a8ca5372bb2192f7bb6a799.svg?invert_in_darkmode" align=middle width=41.002829999999996pt height=24.65759999999998pt/> (c'est à dire à <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.867000000000003pt height=14.155350000000013pt/> pixels), on cherche une perturbation <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode" align=middle width=7.873024500000003pt height=14.155350000000013pt/> de norme minimale telle que :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/174ec305bf2bff7a03422b142ec0864e.svg?invert_in_darkmode" align=middle width=219.58694999999997pt height=49.31553pt/></p>

Dans toute la suite, on utilisera la norme euclidienne. D'autres normes sont évidemment possibles, mais sans amélioration sensible des résultats.

### 1.2 Les attaques adversaires

On cherche un algorithme qui détermine un exemple adversaire à partir d'une image donnée. On dit qu'un tel algorithme réalise une *attaque adversaire*.

Une méthode d'attaque possible est la suivante. Introduisons <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> la fonction qui à une image  associe la probabilité (selon le réseau) que l'image appartienne à la  catégorie <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode" align=middle width=7.113876000000004pt height=14.155350000000013pt/> ; et soit une image <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/179d2870187b498a3d368676c476d2c3.svg?invert_in_darkmode" align=middle width=28.52685pt height=21.683310000000006pt/> de catégorie <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode" align=middle width=7.113876000000004pt height=14.155350000000013pt/>. On cherche alors à minimiser par descente de gradient la fonction <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/412737f1db96536b7345d3593320d88d.svg?invert_in_darkmode" align=middle width=41.118825pt height=22.46574pt/> suivante :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/2ae86d9b14696f8b2ed029353701bd85.svg?invert_in_darkmode" align=middle width=426.95399999999995pt height=49.31553pt/></p>


Cette première fonction est expérimentalement peu satisfaisante car l'attaque échoue souvent : la perturbation reste "bloquée" en <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode" align=middle width=8.219277000000005pt height=21.18732pt/>, et n'évolue pas. Pour pallier cela, on oblige la perturbation à grossir en ajoutant un troisième cas de figure quand <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/d24d41f97b0c6f77006a9ccdba8e3d0e.svg?invert_in_darkmode" align=middle width=157.70320499999997pt height=24.65759999999998pt/>, c'est à dire quand la perturbation n'est pas du tout satisfaisante :
<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/1fc3c244539eacfb16681a07574b36c8.svg?invert_in_darkmode" align=middle width=428.3234999999999pt height=69.041775pt/></p>

Cette deuxième fonction produit presque toujours un exemple adversaire pour un nombre d'étapes de descente de gradient suffisamment élevé (généralement 200 étapes suffisent), et c'est celle-ci qui sera utilisée par la suite.

La Figure 1 montre le résultat d'une attaque adversaire : à gauche l'image originale, au milieu la perturbation et à droite l'image adversaire.

![Résultat d'une attaque adversaire](images/resultat_attaque.png)

On appellera <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/095b66bcf72930ed8c3144d47a46e708.svg?invert_in_darkmode" align=middle width=45.94623pt height=22.46574pt/> la fonction qui à une image associe la perturbation obtenue après <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode" align=middle width=14.999985000000004pt height=22.46574pt/> étapes de descente de gradient (avec un taux d'apprentissage <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/fb9bf6a9b8dad77d5d93ff9b1ed76a45.svg?invert_in_darkmode" align=middle width=63.934695pt height=26.76201000000001pt/>).

### 1.3 Réseau classificateur et base de données utilisé

On réalisera toute cette étude sur un réseau de type <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/61feaa69f87be5f87163b0ef43b33a43.svg?invert_in_darkmode" align=middle width=60.41046pt height=20.09139000000001pt/> (CNN avec Dropout) [2, 3], appliqué au problème de la classification des chiffres manuscrits de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/328aebe52d5ae49b0a885beb672532f3.svg?invert_in_darkmode" align=middle width=43.15047pt height=20.09139000000001pt/>.

La base de données <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/328aebe52d5ae49b0a885beb672532f3.svg?invert_in_darkmode" align=middle width=43.15047pt height=20.09139000000001pt/> est divisée de la manière suivante :

- <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/0461e3b984d62f8868e241fba0ed4c75.svg?invert_in_darkmode" align=middle width=41.09605500000001pt height=21.18732pt/> images d'entraînement (<img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/c18e3ba38c1cec8c110aea91da992706.svg?invert_in_darkmode" align=middle width=43.15047pt height=20.09139000000001pt/>)
- <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/fa35043f335bc43f27e21bc02c268be9.svg?invert_in_darkmode" align=middle width=41.09605500000001pt height=21.18732pt/> images de test (<img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/1cdd767e457f4200598ed2349ab00bc0.svg?invert_in_darkmode" align=middle width=34.52031pt height=18.199830000000013pt/>)
- <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/fa35043f335bc43f27e21bc02c268be9.svg?invert_in_darkmode" align=middle width=41.09605500000001pt height=21.18732pt/> images de validation (<img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/810fb01786c636c9f567304ed653dae0.svg?invert_in_darkmode" align=middle width=25.890315pt height=20.09139000000001pt/>)

Ce réseau est entraîné sur les images de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/c18e3ba38c1cec8c110aea91da992706.svg?invert_in_darkmode" align=middle width=43.15047pt height=20.09139000000001pt/>, et on travaillera systématiquement sur les images de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/1cdd767e457f4200598ed2349ab00bc0.svg?invert_in_darkmode" align=middle width=34.52031pt height=18.199830000000013pt/> par la suite, afin de travailler sur des images que le réseau n'a jamais vues. Les images de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/810fb01786c636c9f567304ed653dae0.svg?invert_in_darkmode" align=middle width=25.890315pt height=20.09139000000001pt/> serviront à évaluer la généralisation des résultats obtenus à de nouvelles images.

Sur les <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/fa35043f335bc43f27e21bc02c268be9.svg?invert_in_darkmode" align=middle width=41.09605500000001pt height=21.18732pt/> images de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/1cdd767e457f4200598ed2349ab00bc0.svg?invert_in_darkmode" align=middle width=34.52031pt height=18.199830000000013pt/>, toutes sauf <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/c2ff8b54320dc3593839122591224746.svg?invert_in_darkmode" align=middle width=16.438455000000005pt height=21.18732pt/> sont classifiées correctement par le réseau.

### 1.4 Quelques résultats

On réalise donc des attaques adversaires sur les images de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/1cdd767e457f4200598ed2349ab00bc0.svg?invert_in_darkmode" align=middle width=34.52031pt height=18.199830000000013pt/>, en effectuant <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/25df05ed4b8476cb9a1a3db76ae8f22c.svg?invert_in_darkmode" align=middle width=24.657765pt height=21.18732pt/> étapes de descente du gradient de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/3a5c9e3056428b808beee92e988ff843.svg?invert_in_darkmode" align=middle width=41.118825pt height=22.46574pt/>, avec un taux d'apprentissage <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/fb9bf6a9b8dad77d5d93ff9b1ed76a45.svg?invert_in_darkmode" align=middle width=63.934695pt height=26.76201000000001pt/>, et on s'intéresse aux valeurs prises par <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/b9a12c4d22f51e9aef9d97ab1b2351a2.svg?invert_in_darkmode" align=middle width=24.31143pt height=24.65759999999998pt/> et <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> au cours de l'attaque.

La Figure 2 a été obtenue en attaquant la première image de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/1cdd767e457f4200598ed2349ab00bc0.svg?invert_in_darkmode" align=middle width=34.52031pt height=18.199830000000013pt/>.

![Attaque adversaire "difficile"](images/attaque_difficile.png){width=60%}

Qualitativement, la norme de la perturbation augmente jusqu'à ce que <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> passe en dessous de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/1c22e0ed21fd53f1f1d04d22d5d21677.svg?invert_in_darkmode" align=middle width=21.004665000000006pt height=21.18732pt/>, à partir de quoi la norme diminue tout en gardant une valeur de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> stabilisée autour de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/358d4d0949e47523757b4bc797ab597e.svg?invert_in_darkmode" align=middle width=21.004665000000006pt height=21.18732pt/>.

Cette image peut être qualifiée de "difficile à attaquer" : il a été nécessaire d'augmenter très fortement la norme de la perturbation pour réussir à casser la prédiction du réseau, ce qui ne se produit qu'après un grand nombre d'étapes, et la norme finale de la perturbation est élevée.

En attaquant une autre image, on a obtenu la Figure 3. Cette image peut alors au contraire être qualifiée de "facile à attaquer" : bien moins d'étapes ont été nécessaires pour casser la prédiction du réseau, la norme finale est très basse, et le pic de très faible amplitude.

![Attaque adversaire "facile"](images/attaque_facile.png){width=60%}

On voit nettement ici l'influence de la valeur du seuil à <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/358d4d0949e47523757b4bc797ab597e.svg?invert_in_darkmode" align=middle width=21.004665000000006pt height=21.18732pt/> dans la fonction <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/f8d449532805c3c7a4839aa3da6af335.svg?invert_in_darkmode" align=middle width=34.566345pt height=22.46574pt/>. Dès que <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> est en dessous de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/358d4d0949e47523757b4bc797ab597e.svg?invert_in_darkmode" align=middle width=21.004665000000006pt height=21.18732pt/>, l'algorithme a pour seul objectif de réduire la norme de la perturbation, et fatalement <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> repasse au dessus de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/358d4d0949e47523757b4bc797ab597e.svg?invert_in_darkmode" align=middle width=21.004665000000006pt height=21.18732pt/>. Il s'agit alors de réduire à la fois <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/b9a12c4d22f51e9aef9d97ab1b2351a2.svg?invert_in_darkmode" align=middle width=24.31143pt height=24.65759999999998pt/> et <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/>, jusqu'à ce que <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> repasse en dessous de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/358d4d0949e47523757b4bc797ab597e.svg?invert_in_darkmode" align=middle width=21.004665000000006pt height=21.18732pt/>, etc.

D'autres exemples d'attaques d'images "faciles" ou "difficiles" à attaquer sont présentés dans l'Annexe A.

Résumons les principales différences qualitatives entre ces deux types d'images :

|                     | Images "faciles" | Images "difficiles" |
| ------------------- | :--------------: | :-----------------: |
| Pic                 | Absent ou faible |        Haut         |
| Étapes nécessaires  |  moins de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/c2c335262ba713d0601ec6d6d01cc102.svg?invert_in_darkmode" align=middle width=16.438455000000005pt height=21.18732pt/>   |    plus de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/88db9c6bd8c9a0b1527a1cedb8501c55.svg?invert_in_darkmode" align=middle width=24.657765pt height=21.18732pt/>    |
| Norme de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode" align=middle width=7.873024500000003pt height=14.155350000000013pt/> finale |      Faible      |       Élevée        |

Pour quantifier plus précisément cette difficulté à attaquer une image, introduisons le concept de *résistance*.

## 2. La résistance <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/b482c509424d573ea9b38bd47c807d5f.svg?invert_in_darkmode" align=middle width=27.96816pt height=22.46574pt/>

### 2.1 La résistance à une attaque

Pour chaque image, on essaie de quantifier la résistance, du réseau à une attaque adversaire. Plusieurs définitions sont possibles, par exemple la norme de la perturbation minimale mettant en échec le réseau :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/e40aa98f986b784f6ceb2386a9adc269.svg?invert_in_darkmode" align=middle width=398.62185pt height=16.438356pt/></p>

Cette expression de la résistance n'est que d'un faible intérêt en pratique, car incalculable. C'est pourquoi on utilisera plutôt les trois définitions suivantes :

- <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/107c8992400f43c183ed2eeb19e4217f.svg?invert_in_darkmode" align=middle width=39.614354999999996pt height=22.46574pt/> la norme finale obtenue après un certain nombre d'étapes dans l'attaque adversaire :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/84d0b1dd50572f095c942463703ada59.svg?invert_in_darkmode" align=middle width=208.1838pt height=16.438356pt/></p>

- <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/72328aeee5790e07313860e0c6f58ed8.svg?invert_in_darkmode" align=middle width=54.21768pt height=22.46574pt/> la hauteur du pic de la norme de la perturbation :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/2dd1b65a55b5507888b1f818b5209f8d.svg?invert_in_darkmode" align=middle width=337.88534999999996pt height=19.726245pt/></p>

- <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/81f5383edfff1a9f81de4a1017d20075.svg?invert_in_darkmode" align=middle width=52.409940000000006pt height=22.46574pt/> le nombre d'étapes qu'il a fallu pour abaisser <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> à <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/358d4d0949e47523757b4bc797ab597e.svg?invert_in_darkmode" align=middle width=21.004665000000006pt height=21.18732pt/> :

<p align="center"><img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/f7f23778fe4807b4c495391947bbb492.svg?invert_in_darkmode" align=middle width=418.29645pt height=19.726245pt/></p>

### 2.2 Une corrélation avec la justesse de la prédiction

Les images attaquées dans l'Annexe A n'ont pas été choisies au hasard : celles du haut sont les 6 premières de de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/1cdd767e457f4200598ed2349ab00bc0.svg?invert_in_darkmode" align=middle width=34.52031pt height=18.199830000000013pt/>, (classifiées correctement par le réseau) , et celles du bas correspondent aux 6 premières erreurs de classification commises par le réseau.

Essayons de voir si ces résultats se généralisent : on étudie la répartition des valeurs de la résistance, d'abord sur <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/7e21c2d99cc4d5b179852d4394bd7121.svg?invert_in_darkmode" align=middle width=24.657765pt height=21.18732pt/> images correctement classifiées (notées **V**), puis sur les <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/c2ff8b54320dc3593839122591224746.svg?invert_in_darkmode" align=middle width=16.438455000000005pt height=21.18732pt/> incorrectement classifiées (notées **F**) :

|                        | <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/107c8992400f43c183ed2eeb19e4217f.svg?invert_in_darkmode" align=middle width=39.614354999999996pt height=22.46574pt/> | <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/72328aeee5790e07313860e0c6f58ed8.svg?invert_in_darkmode" align=middle width=54.21768pt height=22.46574pt/> | <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/81f5383edfff1a9f81de4a1017d20075.svg?invert_in_darkmode" align=middle width=52.409940000000006pt height=22.46574pt/> |
| ---------------------- | :-----: | :---------: | :---------: |
| Premier décile (**V**) | <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/4eb6b36ee9fca4322b72aded3861974e.svg?invert_in_darkmode" align=middle width=29.223975pt height=21.18732pt/>  |   <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/a29a6e201f606b708916e6d177fd915b.svg?invert_in_darkmode" align=middle width=29.223975pt height=21.18732pt/>    |    <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/c1c29f079861ab011226f4979c113a30.svg?invert_in_darkmode" align=middle width=16.438455000000005pt height=21.18732pt/>     |
| Dernier décile (**F**) | <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/858d9e171fde09cd61861b9ae374c95e.svg?invert_in_darkmode" align=middle width=29.223975pt height=21.18732pt/>  |   <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/0cfdcdfdc2f6f93f59b695c6ba8454d0.svg?invert_in_darkmode" align=middle width=29.223975pt height=21.18732pt/>    |    <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/6d25c8494444c5833059403617003de2.svg?invert_in_darkmode" align=middle width=16.438455000000005pt height=21.18732pt/>     |

Dans les trois cas, le premier décile des images correctement classifiées est inférieur au dernier décile des images sur lesquelles le réseau se trompe ! Une corrélation se dessine donc nettement entre la résistance et la justesse de la prédiction du réseau.

## 3. Les corrections adversaires

On observe un autre phénomène : si une attaque adversaire cherche à tromper le réseau, une attaque adversaire sur une image incorrectement classifiée va, le plus souvent, produire une image qui sera correctement classifiée ! On parlera alors de *correction adversaire*.

Toujours avec le même réseau, sur les <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/c2ff8b54320dc3593839122591224746.svg?invert_in_darkmode" align=middle width=16.438455000000005pt height=21.18732pt/> erreurs commises, <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/85cf3a692428195586f3c68fbf98926b.svg?invert_in_darkmode" align=middle width=16.438455000000005pt height=21.18732pt/> des corrections adversaires donnent la bonne catégorie, soit dans <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/064233b1e1dcf749862c000c5dde1e26.svg?invert_in_darkmode" align=middle width=30.137085000000006pt height=24.65759999999998pt/> des cas.

## 4. Une méthode pour réduire l'erreur du réseau ?

### 4.1 Une première méthode...

Exploitons les deux phénomènes précédents pour tenter de réduire l'erreur commise par le réseau : On on détermine la résistance de chaque image du réseau. Si la résistance est supérieure à un certain critère, on considérera que la prédiction du réseau est correcte ; sinon on choisit comme prédiction le résultat de la contre-attaque adversaire.

Sur un lot de 275 images (250 justes, 25 erreurs), avec respectivement <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/aad0bbd7789ee69fbedd1a6506359296.svg?invert_in_darkmode" align=middle width=69.36336pt height=22.46574pt/>, <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/81f5383edfff1a9f81de4a1017d20075.svg?invert_in_darkmode" align=middle width=52.409940000000006pt height=22.46574pt/> et <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/72328aeee5790e07313860e0c6f58ed8.svg?invert_in_darkmode" align=middle width=54.21768pt height=22.46574pt/>, on obtient le nombre d'erreurs commises en fonction du critère choisi, Figure 4.

![Nombre d'erreurs en fonction du critère choisi](images/criteres.png)

Avec des critères à <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode" align=middle width=8.219277000000005pt height=21.18732pt/>, on retrouve naturellement <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/7acc185115b9ffabc044e2079b245b8d.svg?invert_in_darkmode" align=middle width=16.438455000000005pt height=21.18732pt/> erreurs, puisque l'on n'a rien modifié aux prédictions du réseau.

En revanche, avec des critère respectivement à <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/6793dedac02b50b966c382b94ac256d7.svg?invert_in_darkmode" align=middle width=21.004665000000006pt height=21.18732pt/>, <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/7c3a283eba7244935d2cd00c7e8ff476.svg?invert_in_darkmode" align=middle width=37.44312pt height=21.18732pt/> et <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/f2ebeadd36ad2620cbe7f02c861c9da3.svg?invert_in_darkmode" align=middle width=16.438455000000005pt height=21.18732pt/>, le réseau ne commet plus que  <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/7ee94e64f8d5936cc5f263d0ed987bee.svg?invert_in_darkmode" align=middle width=16.438455000000005pt height=21.18732pt/> erreurs.

### 4.2 ...peu efficace à grande échelle.

En appliquand les méthodes précédentes à l'ensemble de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/fa35043f335bc43f27e21bc02c268be9.svg?invert_in_darkmode" align=middle width=41.09605500000001pt height=21.18732pt/> images de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/1cdd767e457f4200598ed2349ab00bc0.svg?invert_in_darkmode" align=middle width=34.52031pt height=18.199830000000013pt/>, on ne réussit qu'à faire passer le nombre d'erreurs de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/c2ff8b54320dc3593839122591224746.svg?invert_in_darkmode" align=middle width=16.438455000000005pt height=21.18732pt/> à <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908740000000003pt height=22.46574pt/> dans le meilleur des cas. Le choix arbitraire d'une fonction de résistance et d'un critère fixé n'est donc pas une méthode efficace ici.

Ceci s'explique simplement : le nombre d'erreurs initial est proportionnellement trop faible (moins de 1%), et donc les faux positifs (images bien classées, mais considérées comme des erreurs par le critère), même si peu fréquents, vont annuler tout le gain obtenu.

### 4.3 Une deuxième méthode, avec un réseau discriminateur

Le choix arbitraire d'un critère et la représentation de la résistance par une seule valeur ne sont donc pas des méthodes efficaces pour réduire l'erreur du réseau. Essayons alors d'affiner la distinction entre les images correctement ou incorrectement prédites.

On s'intéresse ainsi maintenant à toutes les valeurs prises par <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/b9a12c4d22f51e9aef9d97ab1b2351a2.svg?invert_in_darkmode" align=middle width=24.31143pt height=24.65759999999998pt/> et <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> au cours d'une attaque, et on cherche à entraîner un réseau de neurones à faire la distinction entre les images qui seront bien classifiées et celles qui seront mal classifiées.

On retiendra les valeurs sur les étapes multiples de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/7acc185115b9ffabc044e2079b245b8d.svg?invert_in_darkmode" align=middle width=16.438455000000005pt height=21.18732pt/>.

On utilisera un réseau simple, un MLP

(à continuer)

Pour évaluer la généralisation de cette nouvelle méthode, on l'applique sur les images de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/810fb01786c636c9f567304ed653dae0.svg?invert_in_darkmode" align=middle width=25.890315pt height=20.09139000000001pt/>, sur lesquelles on n'a toujours pas travaillé, que ce soit pour l'entraînement du réseau ou la détermination du critère.

(à continuer)

---

## Bibliographie

[1] C. Szegedy, I. Goodfellow & al. CoRR, **Intriguing Properties of Neural Networks.** (Déc. 2013)

[2] A. Krizhevsky, I. Sutskever & G. Hinton. NIPS'12 Proceedings, **ImageNet Classification with Deep Convolutional Neural Networks .** Volume 1 (2012), Pages 1097-1105

[3] N. Srivastava, G. Hinton, A. Krizhevsky & al. JMLP, **Dropout: A Simple Way to Prevent Neural Networks from Overfitting.** Volume 15 (2014), Pages 1929-1958

[4] Trouver un papier qui propose des vraies méthodes d'attaques

## A. Un peu plus de résultats

En Figure 5, les valeurs prises par <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/b9a12c4d22f51e9aef9d97ab1b2351a2.svg?invert_in_darkmode" align=middle width=24.31143pt height=24.65759999999998pt/> et <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/ada5d22dcf9db66090e7c98e48a2196c.svg?invert_in_darkmode" align=middle width=44.68233000000001pt height=22.831379999999992pt/> au cours de l'attaque de 6 images "faciles" à attaquer.

![Attaques adversaires "difficiles"](images/attaques_difficiles.png)

En Figure 6, même chose avec 6 images "difficiles" à attaquer.

![Attaques adversaires "difficiles"](images/attaques_faciles.png)

## B. Calcul pratique de <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/aad0bbd7789ee69fbedd1a6506359296.svg?invert_in_darkmode" align=middle width=69.36336pt height=22.46574pt/>, <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/81f5383edfff1a9f81de4a1017d20075.svg?invert_in_darkmode" align=middle width=52.409940000000006pt height=22.46574pt/> et <img src="https://rawgit.com/maxdinech/car-crash/master/docs/svgs/72328aeee5790e07313860e0c6f58ed8.svg?invert_in_darkmode" align=middle width=54.21768pt height=22.46574pt/>