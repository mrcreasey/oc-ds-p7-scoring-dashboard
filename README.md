# Projet 7 - Implémentez un modèle de scoring

Openclassrooms parcours **Data Scientist**

## Problématique

La société financière, nommée **"Prêt à dépenser"**, propose des crédits à la consommation pour des
personnes ayant peu ou pas du tout d'historique de prêt.

L’entreprise souhaite mettre en œuvre **un outil de “scoring crédit”** pour calculer la qu’un client
rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc
développer **un algorithme de classification** en s’appuyant sur des sources de données variées
(données comportementales, données provenant d'autres institutions financières, etc.).

## Les données

Voici [les données](https://www.kaggle.com/c/home-credit-default-risk/data) pour réaliser le
dashboard. Pour plus de simplicité, vous pouvez les télécharger à
[cette adresse](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip).

## Mission

1. Construire un **modèle de scoring** qui donnera une prédiction sur la probabilité de faillite
   d'un client de façon automatique.
2. Construire **un dashboard interactif** qui montre avec transparence les décisions d’octroi de
   crédit, à destination des gestionnaires de la relation client permettant d'interpréter les
   prédictions faites par le modèle et d’améliorer la connaissance client des chargés de relation
   client.

## Plan du projet

Le plan de ce projet ce trouve en plus de détail dans le document
[project_plan.md](./project_plan.md).

## Livrables de ce projet

### Une dashboard interactif

- répondant aux spécifications ci-dessus et l’API de prédiction du score, déployées chacunes sur le
  cloud.

### Un dossier code

- Le code de la modélisation (du prétraitement à la prédiction)
- Le code générant le dashboard
- Le code permettant de déployer le modèle sous forme d'API

### [Une note méthodologique]

- La méthodologie d'entraînement du modèle
- La fonction coût métier, l'algorithme d'optimisation et la métrique d'évaluation
- L’interprétabilité globale et locale du modèle
- Les limites et les améliorations possibles

### [Un support de présentation]

## Compétences évaluées

- [ ] Utiliser un logiciel de version de code pour assurer l’intégration du modèle
- [ ] Déployer un modèle via une API dans le Web
- [ ] Réaliser un dashboard pour présenter son travail de modélisation
- [ ] Rédiger une note méthodologique afin de communiquer sa démarche de modélisation
- [ ] Présenter son travail de modélisation à l'oral
