import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:

    SECRET_KEY = os.getenv("SECRET_KEY")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    USER_ENABLE_EMAIL = False  # Disable email authentication
    USER_ENABLE_USERNAME = False  # Disable username authentication

    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND")
    timezone = os.getenv("CELERY_TIMEZONE")
    task_serializer = os.getenv("CELERY_TASK_SERIALIZER")
    result_serializer = os.getenv("CELERY_RESULT_SERIALIZER")
    enable_utc = os.getenv("CELERY_ENABLE_UTC")

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):

    DEBUG = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(basedir, "dev.sqlite")


class TestingConfig(Config):

    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(basedir, "test.sqlite")


class ProductionConfig(Config):

    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(basedir, "prod.sqlite")


config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}
