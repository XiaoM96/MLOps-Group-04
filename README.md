# MLOps Group04 2026 Project

Students:
- Julian S194077
- Xiaopeng S194408
- Sharan S242656
- David S250806

## Aim

This project aims to classify EKG signals into four different categories, including Normal, AFib, Other, and Noisy. Users should be able to upload EKG data and receive a classification result indicating the type of signal. This project will utilize the CACHET-CADB (Copenhagen Center for Health Technology - Contextualized Arrhythmia Database) provided by DTU Health. We are using a subset of the dataset that contains only the ECG recordings. It contains 1602 ten-second long ECG samples of AF, NSR, noise, and other rhythm classes, which are manually annotated by two cardiologists. The ECG is sampled at 1024 Hz and a 12-bit resolution. 

## Model

The chosen architecture is EfficientNetBXX, known for its efficiency and performance in image classification tasks. A pretrained model was downloaded from XXX and fine-tuned. 

## Frameworks and Libraries

The project utilizes the following frameworks and libraries:
- Hydra for configuration management
- PyTorch Lightning for model training and evaluation
- Docker for containerization
- UV for python package management


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
