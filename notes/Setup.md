# Setup for a data science project
## Setting up the directory structure

Start with cookiecutter to create a template for your project. This will create a directory with the name of your project and a bunch of files and directories:

`pip install cookiecutter`

`cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science`

Create a new repository on GitHub, copy the URL Push the newly created folder to GitHub:

`git init`

`git add .`

`git commit -m "Initial commit"`

`git branch -M main`

`git remote add origin URL`

`git push --set-upstream origin main`	

## Setting up the environment
Next, install pipenv to create a virtual environment for your project:

`pip install pipenv`	

To enter the virtual environment, run:

`pipenv shell`	

All installed packages will appear in the Pipfile. To install packages, run:

`pipenv install package_name`

Your code needs to be clean and easily readable. Install both a linter and a formatter using the following commands:

`pipenv install --dev pylint black isort`	

You can integrate your formatter and linter into the VSCode Editor. To do so, follow the instructions [here](https://code.visualstudio.com/docs/python/editing) and [here](https://code.visualstudio.com/docs/python/linting). Some instructions they give, you might want to ignore. You can do so using a pyproject.toml file. An example of this file is in the repository.

Yet you might want to run the following command in your command line to see what changes your formatter would make:

`pylint file.py`

`black --diff . | less`
`isort --diff . | less`

To execute the changes on all files, run:

`black .`	

And to see the differences, run:

`git diff file.py`

To make sure your code is reproducible, you should also create a requirements.txt file. This contains all the packages installed in your environment and their versions. To create this file, run:

`pipenv requirements > requirements.txt`

## Setting up the data
Data from online sources is collected using [Windsor.ai](https://www.windsor.ai/). The data is stored in Azure Blob storage. To access the data, you need to install the Azure Storage Blobs client:

`pipenv install azure-storage-blob`	


## Experiment tracking
To track your experiments, you can use [MLflow](https://mlflow.org/). To install it, run:

`pipenv install mlflow`

You can then start the tracking server by running:

`mlflow ui --backend-store-uri sqlite:///mlflow.db`

This automatically creates a database file called mlflow.db. You can then access the tracking server at http://localhost:5000.
In the subfolder 'models', you can find a notebook containing the experiments for the project. This are logged in the mlflow.db file. From these experiments, you can then select the best model and save it to a file. This file can then be used to make predictions on new data.

