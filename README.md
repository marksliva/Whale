# Humpback Whale Identification Challenge
[https://www.kaggle.com/c/whale-categorization-playground](https://www.kaggle.com/c/whale-categorization-playground)

## Python Environment Setup

Set up a conda environment for the project
1. open this project in PyCharm
1. open up settings
1. select `Project:Whale`/`Project Interpreter`
1. click on the gear
1. select conda, new environment with python 3.6
1. click `ok`

## Data Setup
1. in Whale root, `mkdir data`
1. download the files from [https://www.kaggle.com/c/whale-categorization-playground/data](https://www.kaggle.com/c/whale-categorization-playground/data) into `data`
1. unzip `train.zip` and `test.zip`
1. move the train dir to raw train: `mv train raw_train`
1. run the `src/utils/prepare_data.py` script (you can right click on it and click `run`)
  - this will create a directory in data called `train` which will have images in the format of `label/example1.jpg`

## Running tests in Pycharm
1. click on `Edit Configurations:` from the run menu
1. click the `+` button, and then `Python tests/Unittests`
1. name it something like `unit tests`
1. for the `Target/script path` navigate to Whale/tests
1. under pattern, put `*test.py`
1. click on `ok`
1. click on the play button to run the tests

## Running tests from command line
1. create a conda environment using the requirements.txt:
`conda install --file requirements.txt`
1. run the following command:
`python -m unittest discover -s tests -t tests -p *test.py`

## Training and Predicting
Either:
  * right click on `trainer.py` in PyCharm and click the run button
  * run from the command line: `PYTHONPATH=~/PycharmProjects/Whale python src/whale/trainer.py`
