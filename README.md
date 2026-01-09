# MLOps Group04 2026 Project

Students: [fill student ids]:
- Julian
- Xiaopeng
- Sharan
- David 

## Aim

This project aims to classify EKG signals into four different categories, including Normal, AFib, Other, and Noisy. Users should be able to upload EKG data and receive a classification result indicating the type of signal.

## Data

The EKG data is taken from xx, provide by DTU health. Time series data is converted into spectrogram images.

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
