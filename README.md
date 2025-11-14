# PyManyObjetive Framework

## Setup

### Recommended Python version

- This project targets Python 3.9+. Python 3.10 or 3.11 is recommended.
- It is recommended to manage Python versions with [pyenv](https://github.com/pyenv/pyenv).
- The Python version currently used for this project is stored in the .python-version file at the repository root.
- To install and set the project's version:

```bash
$ pyenv install "$(cat .python-version)"
$ pyenv local "$(cat .python-version)"
$ python3 -V
```

### Virtual Python Environment
On the project's root create a virtual python environment.

```bash
$ python3 -m venv ./venv
```

Activate the environment

```bash
$ source ./venv/bin/activate
```

### Install dependencies
Install dependencies using the `requirements.txt` file.

```bash
$ pip install -r requirements.txt
```

## Setup
- [ ] Create my own documentation. 