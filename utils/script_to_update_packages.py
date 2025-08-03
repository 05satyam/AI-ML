import pkg_resources
import subprocess

# Get all installed packages
packages = [dist.project_name for dist in pkg_resources.working_set]

# Upgrade each package
for package in packages:
    subprocess.call(["pip", "install", "--upgrade", package])
