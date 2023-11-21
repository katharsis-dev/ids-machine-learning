from setuptools import setup, find_packages

def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()

def read_file(file):
   with open(file) as f:
        return f.read()
    
VERSION = (0, 0, 1)


# long_description = read_file("../../README.md")
version = ".".join(map(str, VERSION))
requirements = read_requirements("./requirements.txt")

setup(
    name = 'ids-model',
    version = version,
    author = 'Brytton Tsai',
    author_email = 'brytton.tsai.2010@outlook.com',
    url = 'https://nutorus.com',
    description = 'Intrusion Dection Model',

    # long_description_content_type = "plain/text",  # If this causes a warning, upgrade your setuptools package
    # long_description = long_description,

    license = "MIT license",
    packages = find_packages(include=["flaml_flaml_model", "flaml_flaml_model.*"]),  # Don't include test directory in binary distribution
    # packages = find_packages(where="flaml_model"),
    # package_dir = {"": "flaml_model"},
    package_data={
        '': ['saved_models/*'],
    },

    # entry_points = {
    #     "console_scripts": ["train=flaml_model.train:train"]
    # },

    install_requires = requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]  # Update these accordingly
)
