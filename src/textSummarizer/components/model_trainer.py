import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
from textSummarizer.logging import logger
from textSummarizer.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # 1. Load the dataset (which DataTransformation already formatted with prompts)
        dataset = load_from_disk(self.config.data_path)
        
        # 2. Configure 4-bit Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        logger.info(f"Loading Base Model: {self.config.model_ckpt} in 4-bit")
        
        # 3. Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_ckpt,
            quantization_config=bnb_config,
            device_map="auto"
            # trust_remote_code=True # sometimes needed for new models on HF
        )
        
        # 4. Prepare model for PEFT/LoRA
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            task_type=self.config.task_type,
            target_modules="all-linear"
        )
        
        model = get_peft_model(model, lora_config)
        logger.info("Loaded LoRA Config and applied to model.")
        
        # 5. Training Arguments via SFTConfig for trl>=1.0.0
        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            logging_steps=10,
            dataset_text_field="text",
            max_length=512
        )
        
        logger.info("Initializing SFTTrainer (Supervised Fine-Tuning)...")
        # 6. SFT Trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=tokenizer,
            args=training_args,
        )
        
        # 7. Start Training
        logger.info("Starting training loop...")
        trainer.train()
        
        # 8. Save the model locally (Saves ONLY the LoRA adapter weights)
        model.save_pretrained(self.config.output_dir)
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
        logger.info(f"Training complete. LoRA weights saved to {self.config.output_dir}")
