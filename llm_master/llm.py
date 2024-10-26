import torch
import yaml
import argparse
import spacy

from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

torch.cuda.empty_cache()
nlp = spacy.load("en_core_web_sm")

class LLM():
    def __init__(self):
        """
        Initializes the LLM with a specified model and tokenizer from Hugging Face.
        It also logs into Hugging Face using the provided token.
        """
        try:
            self.args = self.config_2_args("llm_master/config.yaml")
            login(token=self.args.hf_login_token)

            # Try loading model and tokenizer from Hugging Face
            self.model_id = self.args.model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        
        except Exception as e:
            print(f"Error loading model from Hugging Face: {e}. Attempting to load a local fallback model.")
            # Fallback to a local model
            self.model_id = "fallback_model"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, local_files_only=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def config_2_args(self, path):
        """ 
        Parses configuration YAML to command-line arguments.
        """
        with open(path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        parser = argparse.ArgumentParser(description="Generate args from config")
        for key, value in yaml_data.items():
            parser.add_argument(f'--{key}', type=type(value), default=value)
        
        args = parser.parse_args([])
        return args

    def parse_player_input(self, player_input):
        """
        Use NLP to understand and classify the player's input.
        :param player_input: The player's input as a string.
        :return: Parsed NLP information, including actions and entities.
        """
        doc = nlp(player_input)

        # Identify important actions (verbs) and entities (nouns, places, characters)
        actions = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        entities = [ent.text for ent in doc.ents]

        return {"actions": actions, "entities": entities}

    def inference(self, prompt):
        """
        Generate text based on the provided prompt using the LLM.
        :param prompt: Text input to generate response.
        :return: Generated text response.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        try:
            output_tokens = self.model.generate(
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
        except Exception as e:
            print(f"Inference failed: {e}")
            return "The response is unavailable at the moment."

    def generate_response(self, quest_name, player_action, player_state, quest_background, npc_memory=None, player_memory=None):
        """
        Generate a story response from the LLM based on the player's action, the quest background, and current state.
        :param quest_name: The name of the current quest.
        :param player_action: The player's input or action.
        :param player_state: Dictionary representing the player's current state.
        :param quest_background: Description of the quest's background.
        :param npc_memory: Optional memory of the NPC being interacted with.
        :param player_memory: Optional memory of the player's recent actions.
        :return: The LLM's generated response for the story.
        """
        prompt = self.generate_prompt(
            quest_name=quest_name,
            player_action=player_action,
            player_state=player_state,
            quest_background=quest_background,
            npc_memory=npc_memory,
            player_memory=player_memory
        )
        
        response = self.inference(prompt)
        return response

    def generate_prompt(self, quest_name, player_action, player_state, quest_background, npc_memory=None, player_memory=None):
        """
        Constructs a prompt incorporating quest details, player actions, and optional memories for a richer context.
        """
        memory_context = f"NPC Memory: {npc_memory}\nPlayer Memory: {player_memory}" if npc_memory or player_memory else ""
        prompt = (f"Quest: {quest_name}\n"
                  f"Background: {quest_background}\n"
                  f"Player Action: {player_action}\n"
                  f"Player State: {player_state}\n"
                  f"{memory_context}\n"
                  f"Respond based on the player's action and the quest's context.")
        
        return prompt

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