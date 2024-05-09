# halu_control
Methods for controlling hallucinations for LLM

# Settings

| MODEL | Strategy | Consistency Rate | Answer Rate | Average Length |
| - | - | - | - | - |
| Mistral-7B-Instruct-v0.1 | - |  90.6 | 98.7 | 96.1 |
| Mistral-7B-Instruct-v0.1 | num_beam = 10 | 95.9 | 99.9 | 144.4 |
| Mistral-7B-Instruct-v0.1 | WhiteListLogits + Greedy | 91.6 | - | - |
| Mistral-7B-Instruct-v0.1 | WhiteListLogits + num_beam = 10 | 94.6 | 100.0 | 155.6 |
| Mistral-7B-Instruct-v0.1 | DoLA + temperature = 0.7 + topP=0.95 | 94.5 | 100.0 | 95.5 | 

