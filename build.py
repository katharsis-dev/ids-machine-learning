import os
import subprocess
import build_configs

def create_virtualenv(folder_path):
    # Navigate to the project directory
    os.chdir(folder_path)

    environment_path = os.path.join(folder_path, "venv")
    if os.path.isdir(environment_path):
        print("Removing existing environment", environment_path)
        subprocess.run(["rm", "-r", "venv"])

    # Create a virtual environment
    subprocess.run(['python', '-m', 'venv', 'venv'])
    print(f"Created Virtual Environment in {folder_path}")

def install_requirements(folder_path):
    # Activate the virtual environment
    activate_script = os.path.join(folder_path, 'venv', 'Scripts' if os.name == 'nt' else 'bin', 'activate')
    pip_location = os.path.join(folder_path, 'venv', 'Scripts' if os.name == 'nt' else 'bin', 'pip')
    python_location = os.path.join(folder_path, 'venv', 'Scripts' if os.name == 'nt' else 'bin', 'python')
    requirements_location = os.path.join(folder_path, "requirements.txt")
    if os.path.isfile(requirements_location):
        # Install requirements
        subprocess.run(f". {activate_script}; cd {folder_path}; pip install --no-deps --force -r requirements.txt", shell=True)
        # subprocess.run(f". {activate_script}; cd {folder_path}; pip install --no-dependencies --force -r requirements.txt", shell=True)
        return True
    print(f"Missing requirements.txt file in {folder_path}")
    return False


def run_pyinstaller(folder_path):
    # Navigate to the project directory
    os.chdir(folder_path)

    # Activate the virtual environment
    activate_script = os.path.join(folder_path, 'venv', 'Scripts' if os.name == 'nt' else 'bin', 'activate')
    pip_location = os.path.join(folder_path, 'venv', 'Scripts' if os.name == 'nt' else 'bin', 'pip')
    python_location = os.path.join(folder_path, 'venv', 'Scripts' if os.name == 'nt' else 'bin', 'python')
    pyinstaller_location = os.path.join(folder_path, 'venv', 'Scripts' if os.name == 'nt' else 'bin', 'pyinstaller')

    script_location = os.path.join(folder_path, 'main.py')

    # Run pyinstaller on main.py
    subprocess.run(f". {activate_script}; cd {folder_path}; pyinstaller --onefile main.py", shell=True)

def process_model_folder(model_folder):
    # Create virtual environment
    create_virtualenv(model_folder)

    # Install requirements
    success = install_requirements(model_folder)

    # Run pyinstaller on main.py
    if success:
        run_pyinstaller(model_folder)
        print(f"Finished Building {model_folder}")

def main():
    # Get the current script directory
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Assume 'models' directory is in the same directory as the script
    models_dir = os.path.join(script_dir, 'models')

    if build_configs.BUILD_FOLDERS:
        # Iterate through each model folder
        for model_folder in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_folder)
            # Check if the item is a directory and not in the list of folders to ignore
            if os.path.isdir(model_path) and model_folder in build_configs.BUILD_FOLDERS:
                process_model_folder(model_path)
    else:
        # List of folder names to ignore
        folders_to_ignore = build_configs.FOLDER_TO_IGNORE

        # Iterate through each model folder
        for model_folder in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_folder)
            # Check if the item is a directory and not in the list of folders to ignore
            if os.path.isdir(model_path) and model_folder not in folders_to_ignore:
                process_model_folder(model_path)

if __name__ == "__main__":
    main()
