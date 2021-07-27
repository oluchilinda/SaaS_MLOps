
This is an MLOps pipeline  for an imaginary SaaS Company that provides a Loan Eligibility Prediction service to Financial institutions like banks, credit unions, savings, and loans associations, investment banks, investment companies, brokerage firms, insurance companies, and mortgage companies.

This guide is for data scientists  ML engineers , software engineers, data analysts who want to apply DevOps principles to ML systems (MLOps). 
MLOps is an ML engineering culture and practice that aims at unifying ML system development (Dev) and ML system operation (Ops) while DevOps is a set of practices that combines software development and IT operations


#### MLOps differs from DevOps in the following ways
- Machine learning is experimental in nature because we would try different algorithms, features, hyperparameters tuning and modelling techniques , unlike software engineering that is static.re

- Testing an ML system is more robust than software systems, in addition to unit and integration testing, we would also be testing model quality , data validation etc

- Deployment for an ML system is more complex as multi step pipelines are set up to continually retrain models and serve productions.

- In production, model decay so faster unlike conventional software systems that do not degrade , you are continously checking for your data statistics , rolling back prediction when data drift occurs.



### FIRST STEPS IN EVERY ML PROCESS
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





##### What I would be implementing the following steps below for MLOps pipeline

- Continuous Integration (CI)
- Continuous Delivery (CD)
- Continuous Training (CT)


### Continous Integration
It’s a practice that allows developers to frequently merge code changes into a central repository where builds and tests then run.
In this setup, a CI pipeline and its components are built, tested, and packaged when new code is committed or pushed to the source code repository.
For the CI setup, I would be implementing the following.

1. Test the feature engineering logic e.g. testsing a function that accepts a categorical data column and encodes it.
2. Test that your model prediction doesn't produce NaN values.
3. Testing that each component in the pipeline produces the expected artifacts.


### Continous Delivery
CD is an extension of continuous integration since it automatically deploys all code changes to a testing and/or production environment after the build stage. 
A  CD pipeline is a set of steps your code changes will go through to make their way to production.
Continuously deliver new pipeline implementations to the target environment that in turn delivers prediction services of the newly trained model. 

For the CD setup, I would implement the following
- Set up infrastructure for model deployment with Terraform (an infrastructure as a code tool)
and also verify the compatibility  of the model with the target infrastructure  such as memory, compute resources before I deploy your model. 
- Validate the data before a batch prediction or retraining.
- Verify that the model meets predictive performance metrics before deploying.
- Test the prediction endpoint service performance such as latency (how long does it take the model to serve a request or batch request.)








### Continous Training
ML models are built on the assumption that the data used in production will be similar to the data observed in the past and  trained our models on but a situation where data is constantly changing , our model performance will be impacted, so our ML models need to be retrained regularly, this is where CT comes in play.

Continuous Training (CT) and is part of the MLOps practice. Continuous training seeks to automatically and continuously retrain the model to adapt to changes that might occur in the data.
While setting up your MLOps for Continous training , the following should be available

1. Feature store : is a data warehouse of features for machine learning and it maintains a list of features along with their definitions and metadata.
A Feature Store is a service that ingests large volumes of data, computes features, and stores them.
<b>Why a Feature Store</b>
    - It manages ineffective and expensive feature engineering practice
    - Discover and reuse available feature sets for their properties, instead of re-creating it
2. Metaadata Pipelines Management :It contains information about the pipeline in order to help with model artifacts.
It will contain the following.
Top-performing models with hyperparameters and metrics.
The path that contains the model artifacts or pickle file
A pointer to the previously trained model if you need to roll back to a previous model version or if you need to produce evaluation metrics for a previous model version when the pipeline is given new test data during the model validation step.


CT answers the following question
- When should the model be retrained or how often should it be retrained? 
The following listed below determine that
    - If your pipeline recieves a data with unexpected features or receiving features with unexpected values. 
    - Significant changes in the statistical properties of data, which means that data patterns are changing, and you need to trigger a retraining of the model to capture these changes.
    - Comparing model performance on models (model's drift)



