from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from textSummarizer.logging import logger
import torch

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_trainer_config()

    def predict(self, text):
        # We load the base model in 4 bit
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        
        logger.info("Attempting to load Trained LoRA adapter for predictions...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_ckpt,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            # Apply your trained LoRA adapter on top
            model = PeftModel.from_pretrained(model, self.config.output_dir)
            logger.info("Successfully merged LoRA Weights!")
        except Exception as e:
            logger.warning("LoRA Weights not found, running Base Instruct Model instead...")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_ckpt,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        
        # Format the user input into the chat prompt
        prompt = f"<|user|>\nSummarize the following conversation:\n\n{text}<|end|>\n<|assistant|>\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate summary
        output = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
        
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        # We want to strip the prompt out and just return the AI's generated response
        summary = decoded_output.split("<|assistant|>")[-1].strip()
        
        return summary
