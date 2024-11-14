---

# Car Price Prediction: End-to-End Machine Learning Project with MLflow and AWS

This repository showcases an end-to-end machine learning project focused on predicting Loan Default using a variety of classification models. The project includes data preprocessing, model training, hyperparameter tuning, deployment, and monitoring—implemented with **MLflow** and **AWS** to ensure reproducibility and scalability.

## Project Highlights

- **Data Source**: Loan Default dataset with features like year, loan_limit,	Gender,	approv_in_adv, and other specifications.
- **ML Pipeline**: Built and managed using MLflow, covering:
  - Data ingestion
  - Data transformation
  - Data validation
  - Model evaluation
  - Model training with various classification algorithms
  - Hyperparameter tuning with MLflow's tracking and logging features
- **Deployment**: Model deployment using AWS services, enabling scalable access and real-time predictions.
- **Model Monitoring**: MLflow and AWS tools used for tracking model performance and updating models as needed.

## Tech Stack

- **MLflow**: Experiment tracking, model versioning, and parameter logging
- **AWS**: Model deployment and monitoring
- **Python**: Main programming language for data handling and model training

---

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py