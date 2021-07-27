from datetime import datetime, timezone
import decimal
import datetime

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

from ..models import TokenBlocklist, Company, Role, CompanyStaff
from .. import db

prediction_main = Blueprint("prediction_main", __name__)


def alchemyencoder(obj):
    """JSON encoder function for SQLAlchemy special classes."""
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    elif isinstance(obj, decimal.Decimal):
        return float(obj)


@prediction_main.route("/make_prediction", methods=["DELETE"])
@jwt_required()
def make_predictions():

    return "hello made predictions"
