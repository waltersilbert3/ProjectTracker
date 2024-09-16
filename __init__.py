# __init__.py
from flask import Flask


# Initialize the Flask app
app = Flask(__name__)


# Import routes after initializing the app
# This ensures the routes.py file can access the app object
from app.routes import *


