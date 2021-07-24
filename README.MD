
This is an MLOps pipeline  for an imaginary SaaS Company that provides a Loan Eligibility Prediction service to Financial institutions like banks, credit unions, savings, and loans associations, investment banks, investment companies, brokerage firms, insurance companies, and mortgage companies.

This guide is for data scientists  ML engineers , software engineers, data analysts who want to apply DevOps principles to ML systems (MLOps). 
MLOps is an ML engineering culture and practice that aims at unifying ML system development (Dev) and ML system operation (Ops) while DevOps is a set of practices that combines software development and IT operations


#### MLOps differs from DevOps in the following ways
- Machine learning is experimental in nature because we would try different algorithms, features, hyperparameters tuning and modelling techniques , unlike software engineering that is static.re

- Testing an ML system is more robust than software systems, in addition to unit and integration testing, we would also be testing model quality , data validation etc

- Deployment for an ML system is more complex as multi step pipelines are set up to continually retrain models and serve productions.

- In production, model decay so faster unlike conventional software systems that do not degrade , you are continously checking for your data statistics , rolling back prediction when data drift occurs.

##### What I would be implementing in this project.

- Continuous Integration (CI)
- Continuous Delivery (CD)
- Continuous Training (CT)


### STEPS 
For any ML project, the following steps would first be implemented

1. Data extraction: You select and integrate the relevant data from various data sources for the ML task.
2. Data analysis: You perform exploratory data analysis (EDA) to understand the available data for building the ML model. 
    - Understanding the data schema and characteristics that are expected by the model.
    - Identifying the data preparation and feature engineering that are needed for the model.
3. Data preparation: The data is prepared for the ML task. 
    - Data cleaning, transformations and feature engineering 
    - Split the data into training, validation, and test sets.
4. Model training: 
    - Different algorithms with the prepared data to train various ML models
    - Hyperparameter tuning to get the best performing ML model.
5. Model evaluation and validation.
    - Model is evaluated on a holdout test set to evaluate the model quality.
    - Model performance is evaluated with metrics to assess quality

