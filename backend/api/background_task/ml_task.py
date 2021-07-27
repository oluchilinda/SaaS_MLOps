
from .. import celery

@celery.task()
def train_model():
    return "hello"