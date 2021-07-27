#### Pip freeze without dependencies of installed packages

```shell
pip install pip-chill
pip-chill > requirements.txt
```

#### Setting up celery
Install a Flask extension to help handle the initialization of Celery:
```shell
$ pip install Flask-Celery-Helper
```

#### Remove unused imports and variables in python file
```shell
pip install autoflake
autoflake --in-place --remove-all-unused-imports example.py
autoflake --in-place --remove-unused-variables example.py
```

#### Set Up Redis
```shell
bash run-redis.sh
```

### Start Celery and flower ( is a tool for monitoring your celery tasks and workers)
```shell
celery -A worker worker --loglevel=INFO
celery flower -A worker worker --loglevel=INFO
```



