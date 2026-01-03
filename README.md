



> **Industrial IoT Analytics** • **Machine Learning** • **MLOps** • **Production Deployment**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) 
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green?logo=fastapi) 
![Docker](https://img.shields.io/badge/Docker-24.0-blue?logo=docker) 
![AWS](https://img.shields.io/badge/AWS-ECS-orange?logo=amazon-aws) 
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2.5-blue?logo=githubactions)


## 📘 Contexte académique
_________________________

Ce projet a été réalisé en vue de l’examen du cours d’Intelligence Artificielle (IA) en Baccalauréat 3 (Bac 3) à
l’Université Protestante de Lubumbashi (UPL).

📌 Encadrement :
M. **Jason MUSA**, Chercheur en Intelligence Artificielle

🎯 Ce travail vise à appliquer de manière pratique les concepts de :

Machine Learning

Intelligence Artificielle

MLOps

Déploiement de modèles en production

## 👤 Auteur

Nom : **ROMMY GERARD**

Niveau : **Baccalauréat 3 (Bac 3)**

Filière : **Intelligence artificielle**

Université : **Université Protestante de Lubumbashi (UPL)**

Cours : **Intelligence Artificielle**

## 🧠 Description du projet

La maintenance prédictive consiste à anticiper les pannes des équipements industriels avant qu’elles ne surviennent, en exploitant les données issues des capteurs.

Ce projet implémente un système intelligent de maintenance prédictive, capable de :

analyser des données de capteurs industriels,

entraîner des modèles de Machine Learning,

prédire les pannes,

et exposer les prédictions via une API.

Le système repose sur une pipeline MLOps complète, automatisée et reproductible, allant de l’ingestion des données jusqu’au déploiement du modèle.

## 🛠 Technical Stack

**Machine Learning & Data** : scikit-learn, pandas, numpy, SMOTE
**API & Validation** : FastAPI, Pydantic
**MLOps & Tracking** : MLflow (expériences & registre de modèles)
**Containerisation** : Docker, Docker Compose
**CI/CD & Automation** : GitHub Actions (build → test → push → deploy)
**Cloud Deployment** : AWS ECR, ECS
**Monitoring** : logs structurés, endpoints de santé, métriques API

## 🎯 Objectifs du projet

Automatiser l’entraînement des modèles ML

Comparer plusieurs algorithmes de classification

Suivre les expériences et versions des modèles

Déployer un modèle prêt pour la production

Appliquer les bonnes pratiques MLOp

## 🏗 Architecture du système

Données → Ingestion → Transformation → Entraînement
→ Évaluation → Modèle *final* → API FastAPI → Production


## 📈 Model Performance

| Model               | Accuracy  | Precision | Recall | F1-Score |
| ------------------- | --------- | --------- | ------ | -------- |
| Random Forest       | **91.2%** | 89.4%     | 92.1%  | 90.7%    |
| Gradient Boosting   | 89.8%     | 87.3%     | 91.5%  | 89.3%    |
| Logistic Regression | 86.4%     | 84.1%     | 88.7%  | 86.3%    |
| SVM                 | 88.1%     | 85.9%     | 90.2%  | 88.0%    |

**Feature Importance:**

1. Tool Wear (32%)
2. Temperature Differential (24%)
3. Torque Variance (21%)
4. Rotational Speed (15%)
5. Equipment Type (8%)
👉 Random Forest a été retenu comme modèle final en raison de ses meilleures performances globales

## 🔁 Pipeline d’entraînement

L’entraînement est entièrement automatisé grâce à une pipeline MLOps, composée de :

Ingestion des données

Transformation & prétraitement

Entraînement des modèles

Évaluation des performances

Sauvegarde du meilleur modèle

Validation des artéfacts

Commande d’entraînement :
*python run_pipeline.py --mode train*
Déploiement avec Docker
*docker-compose up -d --build*
API de prédiction

Une API REST permet d’utiliser le modèle entraîné pour faire des prédictions en temps réel.

Lancer l’API :
*python app.py*


## 🚀 Quick Start

**Clone & Setup**

python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

**Run Pipeline**

```bash
## Train model
python run_pipeline.py --mode train

## Start API
python app.py
```

## 🎓 Conclusion

Ce projet démontre la mise en œuvre pratique d’un système de maintenance prédictive intégrant les concepts clés de :

l’Intelligence Artificielle,

le Machine Learning,

et le MLOps.

Il constitue un travail académique complet, orienté vers des standards professionnels et industriels.
predictive-maintenance-mlops/
│
├── artifacts/           # Modèles, preprocessors, données générées
├── src/                 # Code source
│   ├── components/      # Ingestion, transformation, entraînement
│   ├── pipeline/        # Pipeline MLOps
│   ├── logger.py
│   ├── exception.py
│
├── data/                # Données brutes (ou lien)
├── app.py               # API FastAPI
├── run_pipeline.py      # Lancement entraînement
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── README.md            # ⭐ Exposé principal
└── .gitignore
