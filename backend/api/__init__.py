from datetime import datetime, timedelta, timezone

from flask import Flask
from celery import Celery
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
    set_access_cookies,
    unset_jwt_cookies,
)
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_user import UserManager

from .config import config

db = SQLAlchemy()
celery = Celery()


def create_app(config_name):
    from .models import Company, TokenBlocklist

    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # Setup Flask-User and specify the User data-model

    db.init_app(app)
    Migrate(app, db)
    # celery.init_app(app)
    jwt = JWTManager(app)
    CORS(app)

    from .view.prediction import prediction_main
    from .view.user_views import user_main

    app.register_blueprint(user_main)
    app.register_blueprint(prediction_main)


   

    # Callback function to check if a JWT exists in the database blocklist
    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        jti = jwt_payload["jti"]
        token = db.session.query(TokenBlocklist.id).filter_by(jti=jti).scalar()
        return token is not None

    @app.after_request
    def refresh_expiring_jwts(response):
        try:
            exp_timestamp = get_jwt()["exp"]
            now = datetime.now(timezone.utc)
            target_timestamp = datetime.timestamp(now + timedelta(minutes=30))
            if target_timestamp > exp_timestamp:
                access_token = create_access_token(identity=get_jwt_identity())
                set_access_cookies(response, access_token)
            return response
        except (RuntimeError, KeyError):
            # Case where there is not a valid JWT. Just return the original respone
            return response

    return app
