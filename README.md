# Visual-Question-Answering

## Table of Contents

- [Visual-Question-Answering](#visual-question-answering)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
    - [Loss Graphs](#loss-graphs)
    - [Accuracy Graphs](#accuracy-graphs)
  - [Evaluation](#evaluation)
    - [Evaluation Metrics](#evaluation-metrics)
      - [VizWiz Accuracy](#vizwiz-accuracy)
      - [Answerability](#answerability)
    - [Results](#results)
  - [Deployment](#deployment)
    - [Using Docker](#using-docker)
    - [Using Python](#using-python)
  - [Contributors](#contributors)
  - [References](#references)

## Introduction

## Dataset

## Model Architecture

## Training

### Loss Graphs

### Accuracy Graphs

## Evaluation

### Evaluation Metrics

#### VizWiz Accuracy

#### Answerability

### Results

## Deployment

### Using Docker

### Using Python

1- Make sure to install the requirements

```bash
pip install -r requirements.txt
```

2- Set up the application for the flask

```bash
set FLASK_APP=app.py
```

- If your are using Linux or Mac OS, use the following command instead

```bash
export FLASK_APP=app.py
```

3- Run the following command to start the server

```bash
python -m flask run
```

## Contributors

- [Yousef Kotp](https://github.com/yousefkotp)

- [Adham Mohamed](https://github.com/adhammohamed1)

- [Mohamed Farid](https://github.com/MohamedFarid612)

## References

- [Less Is More: Linear Layers on CLIP Features as Powerful VizWiz Model](https://arxiv.org/abs/2206.05281)

- [CLIP: Connecting text and images](https://openai.com/research/clip)

- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)