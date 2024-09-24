import torch
import yaml
import json
import logging
import argparse
import transformers

from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

torch.cuda.empty_cache()

class LLM():
    def __init__(self, args):
        self.args = self.config_2_args("llm_master/config.yaml")
        login(token=self.args.hf_login_token)

        self.model_id = self.args.model

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)


    def config_2_args(self, path):
        with open(path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        parser = argparse.ArgumentParser(description="Generate args from config")
        for key, value in yaml_data.items():
            parser.add_argument(f'--{key}', type=type(value), default=value)
        
        args = parser.parse_args([])
            
        return args
    

    def inference(self, prompt):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output_tokens = self. model.generate(
                                        inputs['input_ids'],
                                        max_length=self.args.max_length,
                                        num_return_sequences=self.args.num_return_sequences,
                                        temperature=self.args.temperature,
                                        top_k=self.args.top_k,
                                        top_p=self.args.top_p,
                                        do_sample=self.args.do_sample
                                    )

        output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        return output_text
    
    
    def train(self):
        dataset = load_dataset(self.args.dataset)

        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(examples['input'], padding="max_length", truncation=True)
        

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Set up LoRA config
        lora_config = LoraConfig(
                                task_type=TaskType.CAUSAL_LM, 
                                r=int(self.args.r), 
                                lora_alpha=int(self.args.lora_alpha), 
                                lora_dropout=int(self.args.lora_dropout)
                            )
        # Apply LoRA
        peft_model = get_peft_model(self.model, lora_config)


        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.args.training_output,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            num_train_epochs=self.args.num_train_epochs,
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,
            logging_dir=self.args.logging_dir,
            logging_steps=self.args.logging_steps,
        )

        # Define the Trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset["train"]
        )

        # Fine-tune the model
        trainer.train()