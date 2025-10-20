import os
import subprocess

# This codebase works on python 3.14 64-bit

# List of packages to install
packages = [
    "numpy",
    "matplotlib",
    "pandas",
    "opencv-python",
]

# Function to install packages
def install_packages(packages):

    # Upgrade pip to the latest version
    try:
        subprocess.check_call([os.sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("pip has been upgraded successfully.")
    except subprocess.CalledProcessError:
        print("Failed to upgrade pip.")

    # Install each package
    for package in packages:
        try:
            # subprocess.check_call([os.sys.executable, "-m", "pip", "install", package])
            subprocess.check_call([os.sys.executable, "-m", "pip", "install", "--upgrade", package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")

# Run the functions
install_packages(packages)

print("Installation completed.")
