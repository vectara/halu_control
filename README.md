# halu_control
Methods for controlling hallucinations for LLM

# Settings

| MODEL | Strategy | Consistency Rate | Answer Rate | Average Length |
| - | - | - | - | - |
| Mistral-7B-Instruct-v0.1 | ? |  90.6 | 98.7 | 96.1 |
| Mistral-7B-Instruct-v0.1 | Greedy |  93.1 | 99.9 | 113.7 |
| Mistral-7B-Instruct-v0.1 | num_beam = 10 | 95.9 | 99.9 | 144.4 |
| Mistral-7B-Instruct-v0.1 | WhiteListLogits + Greedy | 91.6 | - | - |
| Mistral-7B-Instruct-v0.1 | WhiteListLogits + num_beam = 10 | 94.6 | 100.0 | 155.6 |
| Mistral-7B-Instruct-v0.1 | DoLA + temperature = 0.7 + topP=0.95 | 94.5 | 100.0 | 95.5 | 
| Mistral-7B-Instruct-v0.1 | Greedy + Fava (LLama2-7b) | 93.0 | 99.9 | 113.1 |
| Mistral-7B-Instruct-v0.1 | DPO + LoRA + Greedy | 95.5 | 99.9 | 110.4 |

# Misc

| MODEL | Strategy | Consistency Rate | Answer Rate | Average Length |
| - | - | - | - | - |
| Mistral-7B-Instruct-v0.1 | Greedy |  93.2 | 100.0 | 93.5 | 
| Mistral-7B-Instruct-v0.1 | num_beam = 10 | 95.3 | 100.0 | 127.7 |
| Mistral-7B-Instruct-v0.1 | Greedy + Fava | 93.7 | 100.0 | 93.3 | - |
| Mistral-7B-Instruct-v0.1 | Greedy + DPO(LoRA) | 95.8 | 100.0 | 97.0 | - |