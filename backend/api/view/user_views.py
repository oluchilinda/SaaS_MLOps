from datetime import datetime, timezone
import decimal
import datetime
import os

from flask import (
    Blueprint,
    jsonify,
    request,
)
from flask_jwt_extended import (
    create_access_token,
    get_jwt,
    jwt_required,
)
from werkzeug.security import check_password_hash, generate_password_hash
from flask_user import  roles_required

from ..models import TokenBlocklist, Company, Role

from .. import db

from .prediction import UPLOAD_FOLDER


user_main = Blueprint("user_main", __name__)


def create_token(email):
    access_token = create_access_token(identity=email)
    return jsonify({"msg": "successfully logged in", "data": access_token}), 200


def alchemyencoder(obj):
    """JSON encoder function for SQLAlchemy special classes."""
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    elif isinstance(obj, decimal.Decimal):
        return float(obj)
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        return "Error: Creating directory. " + directory
    
    
@user_main.route("/", methods=["GET"])
def home():
    return  "home page"

@user_main.route("/company", methods=["POST", "GET"])
def company():
    if request.method == "POST":
        email = request.values["email"]
        password = request.values["password"]
        company_name = request.values["company_name"]
        username = request.values["username"]


        if Company.get_by_email(email):
            return {"message": "email already exist"}, 400
        if Company.get_by_username(username):
            return {"message": "username already exist"}, 400

        password = generate_password_hash(password)

        user = Company(email=email, password=password, company_name=company_name, user_name=username)
        # user.roles.append(Role(name='ADMIN'))

        db.session.add(user)
        db.session.commit()
        
        # create company folder
        path = os.path.join(UPLOAD_FOLDER, username)
        createFolder(path)

        return {"msg": "user has been created"}, 201





@user_main.route("/login", methods=["POST"])
def login():
    email = request.values["email"]
    password = request.values["password"]
    
    company_admin = Company.get_by_email(email)
    if company_admin and check_password_hash(company_admin.password, password):
        return create_token(email)
    else:
        {"message": "user doesn't exist"}, 400
        



@user_main.route("/logout", methods=["DELETE"])
@jwt_required()
def logout():
    jti = get_jwt()["jti"]
    now = datetime.now(timezone.utc)
    db.session.add(TokenBlocklist(jti=jti, created_at=now))
    db.session.commit()
    return jsonify(msg="you logged out successfully")





    