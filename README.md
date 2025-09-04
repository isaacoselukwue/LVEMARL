# Pommerman Research

This repository contains research and experiments for the Pommerman environment for my dissertation on multi-agent reinforcement learning.

Python version: 3.12

## Running the Code

Before you run, you need to install the required dependencies. You can do this by running:

```bash
pip install -r requirements.txt
pip install PyYAML orjson gymnasium opencv-python tensorflow pettingzoo pygame pymunk optuna pandas transformers torch accelerate bitsandbytes 
apt update && apt install -y libgl1 libfreetype6-dev libfontconfig1-dev
pip install --upgrade huggingface_hub
hf auth login --token <your_huggingface_token>
```

### Baseline policies:

To run the baseline policies, you can use the following command:

```bash
python -m examples.train --config examples/config/ppo.yml --algo ppo --eval
```

### Custom policies:

To run other policies, you can use the following command:

#### MAPPO Agent Training
To train the MAPPO agent, you can run the following command:
```bash
python -m examples.train --config examples/config/mappo.yml --algo mappo
```

#### QMIX Agent Training
To train the QMIX agent, you can run the following command:
```bash
python -m examples.train --config examples/config/qmix.yml --algo qmix
```

#### HAPPO Agent Training
To train the HAPPO agent, you can run the following command:
```bash
python -m examples.train --config examples/config/happo.yml --algo happo
```

### Hyperparameter Tuning with Optuna
To tune hyperparameters using Optuna, you can run the following command:

#### MAPPO Agent Tuning
To tune hyperparameters for the MAPPO agent, you can run the following command:
```bash
python -m trials.tune_mappo --n-trials=50
```

#### QMIX Agent Tuning
To tune hyperparameters for the QMIX agent, you can run the following command:
```bash
python -m trials.tune_qmix --n-trials=50
```

#### HAPPO Agent Tuning
To tune hyperparameters for the HAPPO agent, you can run the following command:
```bash
python -m trials.tune_happo --n-trials=50
```

#### PPO Ensemble Agent Tuning
To tune hyperparameters for the PPO ensemble agent, you can run the following command:
```bash
python -m trials.tune_ppo_ensemble --n-trials=50
```

#### MAPPO Ensemble Agent Tuning
To tune hyperparameters for the MAPPO ensemble agent, you can run the following command:
```bash
python -m trials.tune_mappo_ensemble --n-trials=50
```

#### HAPPO Ensemble Agent Tuning
To tune hyperparameters for the HAPPO ensemble agent, you can run the following command:
```bash
python -m trials.tune_happo_ensemble --n-trials=50
```

### Ensemble policies:
To run ensemble policies, you can use the following commands:
#### PPO Ensemble Agent Training
To train the PPO ensemble agent, you can run the following command:
```bash
python -m examples.train --config examples/config/ppo_ensemble.yml --algo ppo_ensemble
```

#### MAPPO Ensemble Agent Training
To train the MAPPO ensemble agent, you can run the following command:
```bash
python -m examples.train --config examples/config/mappo_ensemble.yml --algo mappo_ensemble
```

#### HAPPO Ensemble Agent Training
To train the HAPPO ensemble agent, you can run the following command:
```bash
python -m examples.train --config examples/config/happo_ensemble.yml --algo happo_ensemble
```

### Evaluation

#### PPO Agent Evaluation
To evaluate the trained PPO agent, you can run the following command:
```bash
python -m examples.train --config examples/config/ppo.yml --algo ppo --eval
```

#### MAPPO Agent Evaluation
To evaluate the trained MAPPO agent, you can run the following command:
```bash
python -m examples.train --config examples/config/mappo.yml --algo mappo --eval
```

#### QMIX Agent Evaluation
To evaluate the trained QMIX agent, you can run the following command:
```bash
python -m examples.train --config examples/config/qmix.yml --algo qmix --eval
```

#### HAPPO Agent Evaluation
To evaluate the trained HAPPO agent, you can run the following command:
```bash
python -m examples.train --config examples/config/happo.yml --algo happo --eval
```

#### PPO Ensemble Agent Evaluation
To evaluate the trained PPO ensemble agent, you can run the following command:
```bash
python -m examples.train --config examples/config/ppo_ensemble.yml --algo ppo_ensemble --eval
```

#### MAPPO Ensemble Agent Evaluation
To evaluate the trained MAPPO ensemble agent, you can run the following command:
```bash
python -m examples.train --config examples/config/mappo_ensemble.yml --algo mappo_ensemble --eval
```

#### HAPPO Ensemble Agent Evaluation
To evaluate the trained HAPPO ensemble agent, you can run the following command:
```bash
python -m examples.train --config examples/config/happo_ensemble.yml --algo happo_ensemble --eval
```
## LLM Integration
We opted to use LLAMA-3.2 as our language model for LLM-assisted training. This model is designed to provide contextual assistance and improve the decision-making process of our agents. No PEFT techniques were applied, however we generated training data which we can apply later on for self distillation.

## References

Special thanks to the pommerman team for creating the environment and providing a great platform for research. The original repository can be found at [Pommerman GitHub](https://github.com/MultiAgentLearning/playground).
We also extend our gratitude to Meta for providing the Llama-3.2 model, which is used in our LLM-assisted training.
We extend our gratitude to the huggingface team for providing the transformers library, which is used in our LLM-assisted training. The original repository can be found at [Hugging Face Transformers](https://github.com/huggingface/transformers).