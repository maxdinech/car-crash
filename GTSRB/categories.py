# Categories détaillées
vitesse = [0, 1, 2, 3, 4, 5, 6, 7, 8]
depassement = [9, 10, 41, 42]
prioritaire = [12]
passage = [13]
stop = [14]
interdit = [15, 16, 17]
danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
fin = [32]
tournant = [33, 34, 35, 36, 37, 38, 39]
giratoire = [40]

# Couleurs
noir = fin
bleu = tournant + giratoire
rouge = vitesse + depassement + passage + stop + interdit + danger
jaune = prioritaire

# Formes
rond = vitesse + depassement + interdit + fin + tournant + giratoire
losange = prioritaire
hexagone = stop
triangle_haut = passage
triangle_bas = danger
