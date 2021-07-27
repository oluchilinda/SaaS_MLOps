import datetime
import decimal
import json
import os
from datetime import datetime, timezone
from os.path import dirname, join, realpath

from flask import Blueprint, flash, jsonify, redirect, request, current_app
from flask_jwt_extended import (create_access_token, get_jwt, get_jwt_identity,
                                jwt_required)
from flask_user import roles_required
from slugify import slugify
from werkzeug.utils import secure_filename

from .. import db
from ..background_task.ml_task import train_model
from ..models import Company, FeatureName, FileUpload


prediction_main = Blueprint("prediction_main", __name__)


UPLOAD_FOLDER = join(dirname(realpath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {"csv", "xls", "xlsx"}

def alchemyencoder(obj):
    """JSON encoder function for SQLAlchemy special classes."""
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    elif isinstance(obj, decimal.Decimal):
        return float(obj)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@prediction_main.route("/make_prediction", methods=["POST"])
@jwt_required()
def make_predictions():
    # target_variable = request.values['target_variable']
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    uploaded_file = request.files["file"]
    filename = secure_filename(uploaded_file.filename)

    email = get_jwt_identity()
    company = Company.get_by_email(email).user_name
    file_path = os.path.join(UPLOAD_FOLDER, company, filename)

    # check if file already exists before saving
    if filename != "" and allowed_file(uploaded_file.filename):
        try:
            f = open(file_path)
            return {"msg": "you have already uploaded this data for ML training"}
        except (IOError, FileNotFoundError):

            uploaded_file.save(os.path.join(UPLOAD_FOLDER, company, filename))
            file_path = os.path.join(UPLOAD_FOLDER, company, filename)

            save_file = FileUpload(
                file_path=file_path,
                company_id=Company.get_by_email(email).id,
            )
            db.session.add(save_file)
            db.session.commit()

            # do the model training here
            train_model.apply_async(args=[file_path, email])

            return {
                "msg": "model is training ,an email would be sent to you once the job is completed "
            }
    else:
        return {"msg": "invalid file format"}, 400


@prediction_main.route("/feature-store", methods=["GET"])
@jwt_required()
def return_feature_data():
    company_id = Company.get_by_email(get_jwt_identity()).id
    stores = FeatureName.query.filter_by(company_id=company_id).all()
    # current_app.logger.info('%s logged in successfully', stores)

    return jsonify({ 'data': [s.get_feature_scores() for s in stores] })


# @prediction_main.route("/metadata-pipelines", methods=["GET"])
# @jwt_required()
# def metadata_pipelines():
#     companies = Company.query.all()
#     return jsonify({ 'data': [s.get_metadatapipelines() for s in companies] })
