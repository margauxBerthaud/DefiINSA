# DefiIA
 
## Resultats

Pour notre solution nous avons chosit d'implémenter une vectorisation TDIDF et une classification Linear Support VectorClassification. Une selection des paramètre par validation croisée à été faite au préalable afin de déterminer les paramètres les plus efficace pour notre solution. 

Paramètres selectionnées :
 - Norme de penalisation : l2
 - Fonction de perte : hinge, 
 - Paramètre de régularisation : 1
 
Temps d'execution: 
- Nettoyage des données : 645 secondes
- Vectorisation des données (TFIDF) : 49 secondes
- Entrainement du modèle de classification : 124 secondes
- Execution avec des données de test : moins d'une seconde

Notre code a été executé sur nos machine personelle, donc à l'aide d'un CPU. 

## Exigences techniques

Télécharger le script python, il peut être nécessaire de modifier le chemin d'accès des données avant de lancer le script.

Toutes les librairies requises pour le lancement du script sont listées dans le fichier `requirements.txt`
 
 A VOIR:
 
Pour construire un environement fonctionel sur conda, executer les lignes suivantes :
 
 ```
conda create -n Defi python=3.8
conda activate Defi
pip install -r requirements.txt 
python scriptPythonSVC.py
```
