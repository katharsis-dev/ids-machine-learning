import os
import subprocess
import build_configs


# Iterate through each model directory
for model_name, arguments in build_configs.COMMAND_ARGUEMENTS.items():
    model_directory = os.path.join(build_configs.MODEL_FOLDER, model_name)
    executable_path = os.path.join(model_directory, 'dist', 'main')

    # Check if the executable file exists
    if os.path.isfile(executable_path):
        # Build the command to execute
        command = [executable_path] + arguments.split()

        # Execute the command
        subprocess.run(f"{executable_path} {arguments}", shell=True)
    else:
        print(f"Executable not found for model '{model_name}' {executable_path}.")
