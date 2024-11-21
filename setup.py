import subprocess
import sys
import venv
import os
from pathlib import Path

def create_virtual_environment():
    """Create a virtual environment."""
    print("Creating virtual environment...")
    venv.create("venv", with_pip=True)

def get_python_executable():
    """Get the Python executable path based on the operating system."""
    if sys.platform == "win32":
        return os.path.join("venv", "Scripts", "python.exe")
    return os.path.join("venv", "bin", "python")

def install_dependencies():
    """Install project dependencies."""
    python_executable = get_python_executable()
    print("Installing dependencies...")
    subprocess.run([python_executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.run([python_executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    python_executable = get_python_executable()
    nltk_command = (
        "import nltk; "
        "nltk.download('punkt'); "
        "nltk.download('averaged_perceptron_tagger'); "
        "nltk.download('wordnet')"
    )
    subprocess.run([python_executable, "-c", nltk_command])

def download_spacy_model():
    """Download spaCy model."""
    print("Downloading spaCy model...")
    python_executable = get_python_executable()
    subprocess.run([python_executable, "-m", "spacy", "download", "en_core_web_sm"])

def create_directories():
    """Create necessary directories."""
    print("Creating project directories...")
    directories = [
        "uploads",
        "static/sprites",
        "static/backgrounds",
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    """Main setup function."""
    try:
        print("Starting setup...")
        create_virtual_environment()
        install_dependencies()
        download_nltk_data()
        download_spacy_model()
        create_directories()
        print("\nSetup completed successfully!")
        print("\nTo run the application:")
        if sys.platform == "win32":
            print("1. Run: venv\\Scripts\\activate")
        else:
            print("1. Run: source venv/bin/activate")
        print("2. Run: python app.py")
        print("3. Open http://localhost:5000 in your browser")
    except Exception as e:
        print(f"\nError during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
