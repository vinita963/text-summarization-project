import os
from textSummarizer.logging import logger
from datasets import load_dataset, load_from_disk

class DataTransformation:
    def __init__(self, config):
        self.config = config

    def generate_and_tokenize_prompt(self, data_point):
        """
        Unlike Seq2Seq models which need separate 'input_ids' and 'labels', 
        Generative LLMs (Causal LMs) like Phi-3 just need one big string of text.
        We format the dialogue and summary into Phi-3's expected instruction format.
        """
        dialogue = data_point['dialogue']
        summary = data_point['summary']
        
        # Phi-3 specific system/user/assistant token format
        full_prompt = f"<|user|>\nSummarize the following conversation:\n\n{dialogue}<|end|>\n<|assistant|>\n{summary}<|end|>"
        
        return {"text": full_prompt}

    def convert(self):
        # Load data from the downloaded ingestion path
        dataset_samsum = load_from_disk(self.config.data_path)
        
        logger.info("Formatting dataset into Phi-3 Prompts...")
        
        # We apply the formatting to train, test, and validation sets
        # We remove original columns as we only need the standard 'text' column for SFTTrainer
        samsum_pt = dataset_samsum.map(
            self.generate_and_tokenize_prompt,
            remove_columns=['id', 'dialogue', 'summary']
        )
        
        # Save the transformed dataset
        samsum_pt.save_to_disk(os.path.join(self.config.root_dir,"samsum_dataset"))
        logger.info(f"Transformed generative dataset saved at: {os.path.join(self.config.root_dir,'samsum_dataset')}")
