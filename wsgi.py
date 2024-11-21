import sys
import os

# Add your project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from app import app as application

if __name__ == '__main__':
    application.run()
