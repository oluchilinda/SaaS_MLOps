from typing import ByteString
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from flask_user import  UserMixin

from . import db


class BaseModel(db.Model):
    """Define the base model for all other models."""

    __abstract__ = True
    id = db.Column(db.Integer(), primary_key=True)
    created_on = db.Column(db.DateTime(), server_default=db.func.now(), nullable=False)
    updated_on = db.Column(
        db.DateTime(),
        nullable=False,
        server_default=db.func.now(),
        onupdate=db.func.now(),
    )


class TokenBlocklist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    jti = db.Column(db.String(36), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)


class Company(BaseModel):
    __tablename__ = "company"

    email = db.Column(db.String(200), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)
    company_name = db.Column(db.String(200), nullable=False)
    user_name = db.Column(db.String(200), nullable=False)

    company_files = db.relationship("FileUpload", backref="company", lazy=True)
    metadata_pipelines = db.relationship("PipelinesMetadata", backref="company", lazy=True)
    feature_stores = db.relationship("FeatureStore", backref="company", lazy=True)

    roles = db.relationship("Role", secondary="user_roles")

    @classmethod
    def get_by_email(cls, email):
        return cls.query.filter_by(email=email).first()
    @classmethod
    def get_by_username(cls, username):
        return cls.query.filter_by(user_name=username).first()

    @classmethod
    def get_by_id(cls, id):
        return cls.query.filter_by(id=id).first()

    def serialize(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "company_name": self.company_name,
        }





class Role(BaseModel):
    __tablename__ = "roles"
    name = db.Column(db.String(50), unique=True)


class UserRoles(BaseModel):
    __tablename__ = "user_roles"

    company_id= db.Column(
        db.Integer(), db.ForeignKey("company.id", ondelete="CASCADE")
    )
    role_id = db.Column(db.Integer(), db.ForeignKey("roles.id", ondelete="CASCADE"))


class FileUpload(BaseModel):
    __tablename__ = "files"
    file_path = db.Column(db.String(200))
    company_id = db.Column(
        db.Integer(), db.ForeignKey("company.id", ondelete="CASCADE"), nullable=False
    )


class DBConnection(BaseModel):
    pass


class PipelinesMetadata(BaseModel):
    __tablename__ = "pipelines_metadata"
    text = db.Column(db.String(200))
    company_id = db.Column(
        db.Integer(), db.ForeignKey("company.id", ondelete="CASCADE"), nullable=False
    )


class FeatureStore(BaseModel):
    __tablename__ = "feature_store"
    text = db.Column(db.String(200))
    company_id = db.Column(
        db.Integer(), db.ForeignKey("company.id", ondelete="CASCADE"), nullable=False
    )
