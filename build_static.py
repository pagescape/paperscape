import os
import shutil
from flask_frozen import Freezer
from app import app

# Create build directory
build_dir = 'build'
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)
os.makedirs(build_dir)

# Copy static files
shutil.copytree('static', os.path.join(build_dir, 'static'))

# Initialize Frozen-Flask
freezer = Freezer(app)

# Generate static files
freezer.freeze()
