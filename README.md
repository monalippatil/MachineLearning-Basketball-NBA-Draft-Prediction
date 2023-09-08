adv_mla_assignment1
==============================

• Project Overview: The project involves working with the statistical data of the college students playing basketball. In this learning task, the goal is to predict whether a college basketball player will be drafted to join the NBA league based on their statistics.

The annual NBA draft is a highly anticipated event where NBA teams select players from American colleges and international professional leagues to join their rosters. Being drafted to the NBA is a significant achievement for any basketball player, marking a transition to the professional league. This event captures the attention of sports commentators, fans, and enthusiasts who eagerly track the careers of college players and speculate about their chances of being selected by NBA teams.


• Dataset: The dataset provided contains a wide range of features that illuminate players' performance during their college basketball season.
The dataset comprises 64 players' performance attributes, including Games Played (GP), Minutes Played (Min_per), Offensive Rating (ORtg), Defensive Rating (DRtg), Field Goals Made (twoPM), Free Throws Made (FTM), and many others offer insights into various facets of a player's playing style and contribution to their team.

        Basketball Players Metadata: metadata.csv
        Basketball Players Training dataset: train.csv
        Basketball Players Testing dataset: test.csv


• Business Problem: Build a Binary Classification predictive model capable of accuratly determining the likelihood of a college basketball player being drafted into the NBA based on their performance statistics from their records. The model's accurate predictions can provide valuable insights for both players and teams, aiding decision-making during the NBA draft process. Additionally, this model will offer valuable insights to sports commentators, fans, and scouts, aiding them in predicting the potential NBA draft prospects of individual players.

The primary evaluation metric for this task is the AUROC: Area Under the ROC (Receiver Operating Characteristic) Curve.


• The structure of the project directory is as below.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The intermediate and final datasets for modeling.
    │   └── raw            <- The original, immutable dataset that are downloaded from the souce (canvas).
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is <Student Name>-<Student ID>-week<no>_<Task/Model Name> (for ordering),
    │                         the student's initials, id, respective week's work and a short `-` delimited description for task or model, e.g.
    │                         `Patil_Monali_14370946_week2_EDA/Patil_Monali_14370946_week2_RF`.
    │
    ├── references         <- Data dictionaries or metadata manual.
    │
    ├── reports            <- Generated experiment and analysis reports as PDF, Word etc.
    │          
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           
    │   │   └── sets.py    <- Script with function for data cleaning and processing.
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │      │                  predictions
    │      ├── null.py
    │      └── performance.py
    │
    └── pyproject.toml     <- toml file to manage project configurations and dependencies in Python.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


• The subsequent steps to be performed for execution of the project. 

1. Create a new local copy of a remote assignment 1 repository with below command.
    git clone git@github.com:MonaliPatil19/adv_mla_assignment1.git

2. Change the working directory as adv_mla_assignment1.
    cd adv_mla_assignment1

3. Set up a virtual environment, install the required packages using requirement.txt
    pip install -r requirement.txt

3. Install the custom package my-krml-package
   pip install -i https://test.pypi.org/simple/ my-krml-package==0.1.9

4. Execute the EDA notebook for week 2 and week 3. EDA and modeling is in same neetbook for week 1. 
   python Patil_Monali-14370946-week2_EDA.ipynb
   or 
   Patil_Monali-14370946-week3_EDA.ipynb

5. Once EDA for the respective week is completed, execute the modeliing notebook. 
   python Patil_Monali-14370946-week1_LR.ipynb
   or 
   Patil_Monali-14370946-week2_RF.ipynb
   Patil_Monali-14370946-week2_RF_Tuned.ipynb 
   or 
   Patil_Monali-14370946-week3_AdaBoost.ipynb
   Patil_Monali-14370946-week3_AdaBoost_Tuned.ipynb

   Note: The notebooks Patil_Monali-14370946-week*_Tuned.ipynb includes a confusion matrix and the utilization of predict_proba() to estimate the probabilities of target classes on the test dataset, to submitted for the Kaggle competition. 