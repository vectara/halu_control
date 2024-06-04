from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments, pipeline
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import DPOTrainer
import torch
import pandas as pd
import os

if __name__ == "__main__":
    MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
    LORA = True
    GRADIENT_CHECKPOINTING = True

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token 
    model = AutoModelForCausalLM.from_pretrained(MODEL, 
                                                 device_map="auto", 
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2")
    
    dataset = load_dataset("json", data_files="dpo_pairs.jsonl", split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    # Training code
    if LORA:
        peft_config = LoraConfig(
            task_type = TaskType.CAUSAL_LM,
            r = 16,
            lora_alpha = 16,
            lora_dropout = 0.05,
            inference_mode = False,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias='none'
        )
    
    import pdb
    pdb.set_trace()

    args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        evaluation_strategy="steps", 
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps = 100,
        output_dir="./dpo_output",
        optim="paged_lion_8bit",
        learning_rate=1e-6,
        warmup_ratio=0.1,
        lr_scheduler_type="constant_with_warmup",
        # fp16=True,
        bf16=True,
        tf32=True,
        gradient_checkpointing=GRADIENT_CHECKPOINTING)

    dpo_trainer = DPOTrainer(
        model,
        None,
        args=args,
        beta=0.1,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        max_length=2000,
        max_target_length=500,
        max_prompt_length=1500,
        peft_config=peft_config,
    )

    dpo_trainer.train()
