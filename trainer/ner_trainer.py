from .base_trainer import BaseTrainer
from configs.input_config import InputConfig
from transformers import AutoModelForSequenceClassification
from transformers import AdamW, DataCollatorWithPadding
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    pipeline,
    logging,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, PeftModel, TaskType
from accelerate import Accelerator

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from loguru import logger
import evaluate

from trl import SFTTrainer
from datasets import load_dataset, load_metric


class NERTrainer(BaseTrainer):
    def __init__(self, config: InputConfig):
        super().__init__(config)
        self.training_arguements = TrainingArguments(
            output_dir="temp/ner",
            num_train_epochs=self.num_epochs,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim="paged_adamw_32bit",
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            fp16=True,
            bf16=False,
            max_grad_norm=self.max_grad_norm,
            max_steps=self.max_steps,
            warmup_ratio=self.warmup_ratio,
            group_by_length=False,
            lr_scheduler_type=self.lr_scheduler_type,
            report_to="tensorboard",
            remove_unused_columns=False,
        )
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, "float16"),
            bnb_4bit_use_double_quant=False,
        )
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        if self.use_peft:
            self.peft_method = config.model.peft_method
            if self.peft_method == "LoRA":
                self.lora_r = config.model.peft.r
                self.lora_alpha = config.model.peft.alpha
                self.lora_dropout = config.model.peft.dropout
            self.model = self.load_peft_model()
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps
        )
        self.output_dir = "temp/model"
        self.trainer = Trainer(
            model=self.model,
            args=self.training_arguements,
            train_dataset=self.train_data,
            eval_dataset=self.test_data,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.architecture, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.architecture,
            quantization_config=self.bnb_config,
            device_map={"": 0},
            num_labels=1,
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        logger.info("Loaded model")
        return model

    def load_peft_model(self):
        if self.use_peft:
            if self.peft_method == "LoRA":
                self.peft_config = LoraConfig(
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    r=self.lora_r,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"],
                )
            return PeftModel(self.model, self.peft_config)

    def evaluate_fn(self, data_loader):
        self.model.eval()
        all_labels = []
        all_predictions = []
        clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

        with torch.no_grad():
            for batch in data_loader:
                batch_input = self.tokenizer(
                    batch["text"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=200,
                ).to(self.device)
                labels = batch["labels"].unsqueeze(dim=1)
                labels = labels.half().to(self.device)

                outputs = self.model(**batch_input)
                proba = torch.sigmoid(outputs.logits)
                pred_class = torch.where(proba > 0.5, 1, 0)
                clf_metrics.add_batch(references=labels, predictions=pred_class)
            results = clf_metrics.compute()
            logger.info(f"Validation metrics: {results}")
        return results

    def save_adaptor(self):
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    def train(self):
        self.trainer.train()
