import os
from api import create_app

app = create_app(os.getenv("FLASK_ENV"))

