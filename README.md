# halu_control
Methods for controlling hallucinations for LLM in Summarization

# Benchmark settings

| MODEL | Strategy | Consistency Rate | Answer Rate | Average Length |
| - | - | - | - | - |
| Mistral-7B-Instruct-v0.1 | Greedy |  93.2 | 100.0 | 93.5 | 
| Mistral-7B-Instruct-v0.1 | num_beam = 10 | 95.3 | 100.0 | 127.7 |
| Mistral-7B-Instruct-v0.1 | Greedy + DoLA | 93.7 | 100.0 | 93.6 |
| Mistral-7B-Instruct-v0.1 | Greedy + DPO(LoRA) | 95.8 | 100.0 | 97.0 |
| Mistral-7B-Instruct-v0.1 | Greedy + Fava | 93.7 | 100.0 | 93.3 |
| Mistral-7B-Instruct-v0.1 | DPO(LoRA) + num_beam=10 | 96.9 | 100.0 | 123.7 |

Note: Prompt slightly different from the orginal [HHEM](https://huggingface.co/spaces/vectara/leaderboard) benchmark, causing different numbers.

# How to reproduce the experiments
1. Download the leaderboard dataset (https://huggingface.co/spaces/vectara/leaderboard/raw/main/src/datasets/leaderboard_dataset.csv)
2. Generate the model response ``generated.csv``, see methods below
3. Run evaluation on the reposnse file
```bash
python -c "from leaderboard import run_eval;run_eval('generated.csv')"
```

# Methods

## Baseline
- Greedy/Beam Search
- Notebook: [1_decoding.ipynb](1_decoding.ipynb)

## DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models
- [Paper Link](https://arxiv.org/abs/2309.03883)
- Notebook: [2_dola.ipynb](2_dola.ipynb)

## Fine-tuning Language Models for Factuality
- [Paper Link](https://arxiv.org/abs/2311.08401)
- Notebook: [3_dpo.ipynb](3_dpo.ipynb)
- Training code: [dpo_training.py](dpo_training.py)
- Note: our setting is different from the original paper, we used CNN/Dailymail+XSum+VitaminC as the source dataset and [HHEM](https://huggingface.co/vectara/hallucination_evaluation_model) model as the reference metric for factuality.

## Fine-grained Hallucination Detection and Editing For Language Models
- [Web Link](https://fine-grained-hallucination.github.io/)
- Notebook: [4_fava.ipynb](4_fava.ipynb)