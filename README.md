# DIY-MKG
This repository contains the source code for the EMNLP 2025 Demo submission "DIY-MKG: An LLM-Based Polyglot Language Learning System".

## Introduction

Existing language learning tools, even those powered by Large Language Models (LLMs), often lack support for polyglot learners to build linguistic connections across vocabularies in multiple languages, provide limited customization for individual learning paces or needs, and suffer from detrimental cognitive offloading.
To address these limitations, we design Do-It-Yourself Multilingual Knowledge Graph (DIY-MKG), an open-source system that supports polyglot language learning.
DIY-MKG allows the user to build personalized vocabulary knowledge graphs, which are constructed by selective expansion with related words suggested by an LLM. The system further enhances learning through rich annotation capabilities and an adaptive review module that leverages LLMs for dynamic, personalized quiz generation. In addition, DIY-MKG allows users to flag incorrect quiz questions, simultaneously increasing user engagement and providing a feedback loop for prompt refinement.
Our evaluation of LLM-based components in DIY-MKG shows that vocabulary expansion is reliable and fair across multiple languages, and that the generated quizzes are highly accurate, validating the robustness of DIY-MKG. 

More introduction can be found in our paper.

## Installation

Please use the following 3 steps to initialize the environment.

1. `conda create -n DIY-MKG python=3.10.18`

2. `conda activate DIY-MKG`

3. `pip install -r requirements.txt`

Please do not hesitate to contact the repository owner if you need additional assistance.

## Usage

First, activate the environment by `conda activate DIY-MKG`.

Then, update the configuration file `config.json` to intialize the interface. Feel free to keep the default values.
- `QUIZ_MCQ_COUNT` specifies the number of multiple-choice questions in each quiz.
- `QUIZ_FILL_IN_BLANK_COUNT` specifies the number of fill-in-the-blank questions in each quiz.
- `provider` specifies whether an API-based LLM or a local LLM is used in the interface. Currently, the value can either be `openai` or `local`.
- `openai_model` specifies the OpenAI model name used in the interface.
- `api_key_env_var` specifies the environment variable for the OpenAI API key. Please set up the environment variable properly before starting the interface.
- `local_model_name` specifies the name of the local model to be used.
- `local_model_directory` specifies the directory where the local model is saved.

Finally, use `python app.py` to start the interface.

## Evaluation

The evaluation scripts and data can be found [here](https://github.com/kenantang/DIY-MKG-evaluation).