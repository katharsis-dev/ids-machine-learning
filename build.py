import os
import subprocess
from constants import IMPORT_PACKAGE 

def create_virtualenv(folder_path):
    # Navigate to the project directory
    # os.chdir(folder_path)

    environment_path = os.path.join(folder_path, "venv")
    if os.path.isdir(environment_path):
        print("Removing existing environment", environment_path)
        subprocess.run(f"rm -r {environment_path}", shell=True)

    # Create a virtual environment
    # subprocess.run(['python', '-m', 'venv', 'venv'])
    subprocess.run(f"cd {folder_path}; python -m venv venv;", shell=True)
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

def install_package(folder_path):
    # Navigate to the project directory
    # os.chdir(folder_path)

    # Activate the virtual environment
    activate_script = os.path.join(folder_path, 'venv', 'Scripts' if os.name == 'nt' else 'bin', 'activate')
    pip_location = os.path.join(folder_path, 'venv', 'Scripts' if os.name == 'nt' else 'bin', 'pip')
    python_location = os.path.join(folder_path, 'venv', 'Scripts' if os.name == 'nt' else 'bin', 'python')
    setup_location = os.path.join(folder_path, 'setup.py')

    if os.path.isfile(setup_location):
        # Run pyinstaller on main.py
        subprocess.run(f". {activate_script}; cd {folder_path}; pip install .", shell=True)

        print("Finished installing module")
        return True
    print(f"Missing setup.py file.")



def select_model(models_dir) -> str:
    """
    Returns the selected model path
    """
    model_paths = os.listdir(models_dir)
    print("Select one of the following models to build:\n")
    for i in range(1, len(model_paths) + 1, 1):
        print(f"{i}. {model_paths[i - 1]}")
    print("")

    model_index = -1
    while model_index < 0 or model_index >= len(model_paths):
        try:
            model_index = int(input("Model Number: ")) - 1
            print("")
        except TypeError:
            pass
        except ValueError:
            pass

    return model_paths[model_index]

def build_environment(model_package_path):
    create_virtualenv(model_package_path)
    install_requirements(model_package_path)

    install_package(model_package_path)

def update_run_script(model_name):
    import_package_command = f"from {model_name}.model import Model"

    if os.path.isdir(IMPORT_PACKAGE):
        subprocess.run(["rm", IMPORT_PACKAGE])

    with open(IMPORT_PACKAGE, "w") as file:
        file.write(import_package_command)
    print(f"Run Script Import Module Updated: {import_package_command}\n")

def activate_new_environment(folder_path):
    activate_script = os.path.join(folder_path, 'venv', 'Scripts' if os.name == 'nt' else 'bin', 'activate')

    message_title = "New Python Environment Created"
    print("=" * 30, message_title, "=" * 30, "\n")
    print("Run following command to activate before runing run.py:")
    print(f"COMMAND: source {activate_script}\n")
    print("=" * (60 + len(message_title) + 2), "\n")
    # activate = input("Activate newly created environment (Y/N)? ").lower()
    # if activate == "y" or activate == "yes":
    #     subprocess.run(f"/bin/bash source {activate_script}", shell=True, check=True)

def test_build(run_file_path, environment_path):
    # Activate the virtual environment
    activate_script = os.path.join(environment_path, 'venv', 'Scripts' if os.name == 'nt' else 'bin', 'activate')
    python_location = os.path.join(environment_path, 'venv', 'Scripts' if os.name == 'nt' else 'bin', 'python')

    subprocess.run(f". {activate_script}; cd {run_file_path}; python run.py --test", shell=True)


def main():
    original_directory = os.getcwd()

    # Get the current script directory
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Assume 'models' directory is in the same directory as the script
    models_dir = os.path.join(script_dir, 'models')

    # Prompt the user for model selection for build
    selected_model = select_model(models_dir)
    model_package_path = os.path.join(models_dir, selected_model)
    
    # Build the Python Virtual Environment
    build_environment(model_package_path)

    # Create a file so the run.py knows what command to use to import the required package
    update_run_script(selected_model)

    # Test Build Results
    test_build(original_directory, model_package_path)

    # Activate environment
    activate_new_environment(model_package_path)


if __name__ == "__main__":
    main()

