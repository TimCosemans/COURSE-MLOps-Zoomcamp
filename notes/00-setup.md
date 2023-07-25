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

You can integrate your formatter and linter into the VSCode Editor. To do so, follow the instructions [here](https://code.visualstudio.com/docs/python/editing) and [here](https://code.visualstudio.com/docs/python/linting). Some instructions they give, you might want to ignore. You can do so using a pyproject.toml file. 

<pre>
<code>
[tool.pylint.messages_control]

disable = [
    "missing-function-docstring",
    "missing-final-newline",
    "missing-class-docstring",
    "missing-module-docstring",
    "invalid-name",
    "too-few-public-methods"
]

[tool.black]
line-length = 88
target-version = ['py39']
skip-string-normalization = true

[tool.isort]
multi_line_output = 3
length_sort = true

</code>
</pre>

You can also disable notifications for certain classes by adding 

<pre>
<code>
`# pylint: disable=too-dew-public-methods`
</code>
</pre>

Error code returned will not be 0 if there are warnings. So make sure to disable all warnings you do not want to see.
Yet you might want to run the following command in your command line to see what changes your formatter would make:

`pylint file.py`

`black --diff . | less #to format code`
`isort --diff . | less #to sort imports`

To execute the changes on all files, run:

`black .`	
`isort .`	

And to see the differences, run:

`git diff file.py`

To make sure your code is reproducible, you should also create a requirements.txt file. This contains all the packages installed in your environment and their versions. To create this file, run:

`pipenv requirements > requirements.txt`

