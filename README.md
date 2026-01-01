# 🏥 Plateforme de Diagnostic ML - Cancer du Sein

## 📄 Description
Ce projet vise à améliorer la détection précoce du cancer du sein en utilisant des techniques de Machine Learning (ML) et de Deep Learning (DL). Il propose une solution complète allant de l'analyse exploratoire des données à une application web interactive pour les patients, les médecins et les administrateurs.

Les méthodes traditionnelles (biopsies) étant parfois lentes et coûteuses, cette plateforme permet d'analyser rapidement les caractéristiques cellulaires issues de la méthode FNA (Fine Needle Aspiration) pour fournir une aide au diagnostic fiable.

## 📂 Structure du Projet

Le répertoire est organisé en deux parties principales :

### 1. Analyse et Modélisation (Racine)
*   **`BreastCancer.ipynb`** & **`MLProject.ipynb`** : Notebooks Jupyter contenant l'analyse exploratoire des données (EDA), le pré-traitement, l'entraînement des modèles et l'évaluation des performances.
*   **`data.csv`** : Le jeu de données médicales utilisé pour l'entraînement.
*   **`saved_model/`** : Contient les modèles entraînés (ex: Keras, Joblib) prêts à être utilisés.

### 2. Application Web (`breast_cancer_ml_platform/`)
Une application **Flask** complète qui déploie les modèles pour une utilisation en conditions réelles.
*   **`app.py`** : Point d'entrée de l'application web.
*   **`templates/`** : Interfaces utilisateurs (Dashboards Admin, Docteur, Patient).
*   **`models/`** : Modèles spécifiques utilisés par l'application web.

## 🎯 Objectifs et Fonctionnalités

Le projet implémente 3 objectifs métiers (Business Objectives) :

*   **BO-1 : Détection Rapide** - Prédiction automatique (Malin/Bénin) avec une haute précision.
*   **BO-2 : Explicabilité** - Transparence des prédictions pour aider à la décision médicale (Feature Importance).
*   **BO-3 : Stratification des Risques** - Classification des niveaux de risque (Faible, Moyen, Élevé) pour prioriser les soins.

### Rôles Utilisateurs
*   **👤 Patient** : Soumettre ses données médicales, consulter son diagnostic simplifié et son niveau de risque.
*   **👨‍⚕️ Médecin** : Accéder aux prédictions détaillées, analyser l'importance des caractéristiques cliniques et gérer les dossiers patients.
*   **🛠️ Admin** : Gérer les utilisateurs, superviser les modèles ML et la sécurité du système.

## 🚀 Installation et Démarrage

### Prérequis
*   Python 3.8+
*   pip

### Lancer l'Application Web

1.  Accédez au dossier de la plateforme :
    ```bash
    cd breast_cancer_ml_platform
    ```

2.  Installez les dépendances nécessaires :
    ```bash
    pip install -r requirements.txt
    ```

3.  Lancez le serveur :
    ```bash
    python app.py
    ```
    L'application sera accessible via : **http://localhost:5000**

### Explorer les Notebooks
Pour visualiser l'analyse de données et le processus d'entraînement, ouvrez `BreastCancer.ipynb` ou `MLProject.ipynb` directement dans VS Code ou Jupyter.

## 👥 Auteurs
*   Projet PIDEV
