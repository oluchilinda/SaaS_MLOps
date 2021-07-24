

### MLOps Workflow
This is a simple guide to integrating MLOps principles into an anomaly detector model based on the credit card dataset
We utilize MLFlow to reap the benefits of MLOps with just a few lines of code.

MLFlow is an API that allows you to integrate MLOps principles into your projects with minimal
changes made to existing code.
##### With MLFlow you can do the following
1. Create experiments: Experiments in MLFlow
essentially allow you to group your models and any
relevant metrics. 
2. Model and metric logging: MLFlow allows you to
save a model  training, testing, and validation in a modularized form and log all of the
metrics related to the model run. 
3. Compare model metrics: MLFlow also allows you
to compare different models and their metrics all at
once. And so, when performing validation to help
tune a model’s hyperparameters, you can compare all
of the selected metrics together in MLFlow using its
user interface. 
4. Model Registry: MLFlow also adds functionality to
allow you to implement a model registry, allowing you
to define what stage a particular model is in. 
5. Deploy on Cloud Services :  MLFlow also allows you to deploy
your models on popular cloud services such as
Amazon SageMaker, Microsoft Azure, Google Cloud,
and Databricks.


###  Set Up Local System
1.Set up virtual enviroment with venv
2. Set up virtual env for your jupyter notebook with ipykernel

```shell
mlflow ui -p 1234
```

Output:
```shell
(env) oluchi@apples-Air-5 MLOps % mlflow ui -p 1234      
[2021-07-24 08:18:57 +0100] [7272] [INFO] Starting gunicorn 20.1.0
[2021-07-24 08:18:57 +0100] [7272] [INFO] Listening at: http://127.0.0.1:1234 (7272)
[2021-07-24 08:18:57 +0100] [7272] [INFO] Using worker: sync
[2021-07-24 08:18:57 +0100] [7273] [INFO] Booting worker with pid: 7273
```



![Dimension and Fact Table!](/images/dimension_facts.png "Dimension and Fact Table")