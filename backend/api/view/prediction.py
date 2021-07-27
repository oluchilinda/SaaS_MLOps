import datetime
import decimal
import os
from datetime import datetime, timezone
from os.path import dirname, join, realpath

from flask import Blueprint, flash, jsonify, redirect, request
from flask_jwt_extended import (
    create_access_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
)
from flask_user import roles_required
from slugify import slugify
from werkzeug.utils import secure_filename

from .. import db
from ..models import Company, FileUpload
from ..background_task.ml_task import train_model


prediction_main = Blueprint("prediction_main", __name__)


UPLOAD_FOLDER = join(dirname(realpath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {"csv", "xls", "xlsx"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        return "Error: Creating directory. " + directory


@prediction_main.route("/make_prediction", methods=["POST"])
@jwt_required()
def make_predictions():
    # target_variable = request.values['target_variable']
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    uploaded_file = request.files["file"]
    filename = secure_filename(uploaded_file.filename)

    company = Company.get_by_email(get_jwt_identity()).user_name
    file_path = os.path.join(UPLOAD_FOLDER, company, filename)

    # check if file already exists before saving
    if filename != "" and allowed_file(uploaded_file.filename):
        try:
            f = open(file_path)
            return {"msg": "you have already uploaded this data for ML training"}
        except FileNotFoundError:

            uploaded_file.save(os.path.join(UPLOAD_FOLDER, company, filename))
            file_path = os.path.join(UPLOAD_FOLDER, company, filename)

            save_file = FileUpload(
                file_path=file_path,
                company_id=Company.get_by_email(get_jwt_identity()).id,
            )
            db.session.add(save_file)
            db.session.commit()

            # do the model training here
            train_model.apply_async()

            return {
                "msg": "model is training ,an email would be sent to you once the job is completed "
            }
    else:
        return {"msg": "invalid file format"}, 400
