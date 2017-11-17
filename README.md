# Car Crash

    « No stop signs,
      Speed limit,
      Nobody's gonna slow me down »
    
    -- **AC/DC**, *Highway to Hell*

---

*L’efficacité exceptionnelle des réseaux de neurones les rend intéressant dans de nombreux domaines, en particulier celui de la conduite autonome. Mais est-ce  bien raisonnable de leur faire autant confiance ?*

## Thème -- Milieux : interactions, interfaces, homogénéité, ruptures

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
- [x] Même chose en utilisant *PyTorch*.
- [x] Étudier le fonctionnement des réseau neuronaux convolutifs (***CNN***).
- [x] Mettre en place un CNN avec *PyTorch* pour la reconaîssance de *MNIST*.

### Partie I.B (Classification des panneaux : *GTSRB*)

- [ ] Mettre en place un *CNN* avec *PyTorch* pour la classification des panneaux (Réseau *A*).
- [ ] Trouver un réseau de classification des panneaux utilisé dans des voitures du commerce (Réseau *B*).
- [ ] Optionnel : Faire fonctionner ces réseaux dans une application iOS pour avoir des résultats en temps réel.

### Partie II (Attaque des réseaux : *Dodging* et *Impersonating*)
- [ ] Sur les réseaux collectés (*A* et *B*), essayer de tromper la reconaîssance de manière "naïve" (par ex. descente de gradient sur un panneau.)
- [ ] Étudier l'état de l'art dans ce domaine : examples adversaires et attaques *black-box*.
- [ ] Mettre en place ces attaques sur les trois réseaux.
- [ ] Étudier et comparer les résultats.

### Partie III (Renforcement des réseaux)
- [ ] Inventer et mettre en place un renforcement "naïf" contre de telles attaques.
- [ ] Étudier le fonctionnement des réseaux adversaires génératifs (***GANs***) comme technique de renforcement possible.
- [ ] Collecter et expérimenter d'autres stratégies de renforcement existantes.
- [ ] Comparer le succès des attaques sur les trois réseaux, avec diverses techniques de renforcement mises en place. 
- [ ] Conclure

---


## Liens utiles

- [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb)
- [Arxiv Sanity Preserver (Recherche de publications)](http://www.arxiv-sanity.com)
- [Glossaire complet et détaillé](http://www.wildml.com/deep-learning-glossary/)
- [Comparaison des différents GANs](https://github.com/znxlwm/pytorch-generative-model-collections)

## Dépendances

- Python 3
- numpy, matplotlib, PyTorch (torch), scikit-image, pandas, glob, joblib, wget, zipfile, shutil, tqdm
