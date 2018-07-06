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

## Running tests from the command line
1. create a conda environment using the requirements.txt:
`conda install --file requirements.txt`
1. run the following command (outputs to TestOutput/log and git adds it):
`./run-tests.sh`

## Training and Predicting
Either:
  * right click on `trainer.py` in PyCharm and click the run button
  * run from the command line: `PYTHONPATH=~/PycharmProjects/Whale python src/whale/trainer.py`

## Data Exploration
```
train = pd.read_csv('data/train.csv')
>>> train.groupby('Id').nunique().sort_values('Image', ascending=False)
             Image Id
Id
new_whale    810   1
w_1287fbc     34   1
w_98baff9     27   1
w_7554f44     26   1
w_1eafe46     23   1
w_fd1cb9d     22   1
w_ab4cae2     22   1
w_693c9ee     22   1
w_987a36f     21   1
w_43be268     21   1
w_73d5489     21   1
w_f19faeb     20   1
w_95874a5     19   1
w_9b401eb     19   1
...

train.groupby('Id').nunique().sort_values('Image', ascending=False).describe()
             Image      Id
count  4251.000000  4251.0
mean      2.317102     1.0
std      12.586066     0.0
min       1.000000     1.0
25%       1.000000     1.0
50%       1.000000     1.0
75%       2.000000     1.0
max     810.000000     1.0
```
