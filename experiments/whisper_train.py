"""
Whisper Model Fine-tuning Script
==================================
Trains the OpenAI Whisper-Large-V3-Turbo model on custom audio dataset for prescription audio recognition.
Uses the 'yezarniko/medicines' dataset, implements gradient checkpointing for memory efficiency,
and supports multiple devices (CUDA with bfloat16, MPS, CPU). Configures training arguments for multi-GPU
distributed training and saves the fine-tuned model with safetensors format.
"""

import torch
import os
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union

def main():
    # 1. Hardware & Device Setup
    # Automatically selects CUDA (NVIDIA GPU) if available, otherwise Apple Silicon (MPS), otherwise CPU
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"Using device: {device}")
    
    # Using whisper-large-v3-turbo
    model_id = "openai/whisper-large-v3-turbo"

    # 2. Load Processor and Model
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load model with bfloat16 for A100 memory/speed efficiency
    # If bf16 is unavailable on testing machines, it will fallback or torch will raise an error.
    # Since we are deploying to A100s, bf16 is natively supported.
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        low_cpu_mem_usage=True, 
        use_safetensors=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32
    )
    
    # Move generation parameters to generation_config to avoid Trainer crash on save_pretrained
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    
    # Enable gradient checkpointing to save VRAM and allow larger batch sizes
    model.gradient_checkpointing_enable()
    
    # 3. Load and Prepare Dataset
    print("Loading dataset 'yezarniko/medicines'...")
    dataset = load_dataset("yezarniko/medicines")
    
    # Set the audio column to resample to 16kHz on the fly
    dataset = dataset.cast_column("file", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        # Load and resample audio data
        audio = batch["file"]

        # Compute log-Mel input features from input audio array 
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # Encode target text to label ids 
        # Note: adjust this if the text column is named differently in 'yezarniko/medicines'
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    print("Extracting features from dataset... (This may take a while)")
    # If the dataset has "train", use it. Otherwise map over the dataset dict.
    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    
    # Keep only a small subset for testing purposes if you just want to verify the script.
    # To run on full data, remove the `select(range(...))` part.
    encoded_dataset = dataset[split_name].map(
        prepare_dataset, 
        remove_columns=dataset[split_name].column_names,
        num_proc=1
    )
    
    # 4. Custom Data Collator
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # Replace padding with -100 to ignore it when calculating loss
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # If bos token is appended in previous tokenization step, cut it here
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch
            
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # 5. Setup Trainer
    # NOTE: To train properly, you'd likely want to increase max_steps or use num_train_epochs.
    # The below settings are geared towards running smoothly without memory crashing on the 16GB Mac.
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-turbo-finetuned",
        per_device_train_batch_size=16,         # Increased batch size for 80GB VRAM
        gradient_accumulation_steps=2,          # Effective batch size of 32
        learning_rate=1e-5,                     # Lower learning rate for full fine-tuning vs LoRA
        warmup_steps=500,                       # Longer warmup 
        max_steps=4000,                         # Substantially more steps for full training
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=25,
        report_to="none",                       # Change to "wandb" or "tensorboard" if you want to track metrics
        remove_unused_columns=False,
        label_names=["labels"],
        optim="adamw_torch",
        predict_with_generate=False,            # Set to True and add compute_metrics for WER
        bf16=True,                              # Enable BFloat16 native to A100 GPUs
        fp16=False,
        dataloader_num_workers=8,               # Exploit the 32-core Xeon processor for fast I/O
    )
    
    # Create an evaluation split if there's only a train split
    split_dataset = encoded_dataset.train_test_split(test_size=0.1, seed=42)
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final specialized model...")
    trainer.save_model("./whisper-turbo-finetuned-final")
    print("Training job complete!")

if __name__ == "__main__":
    main()
