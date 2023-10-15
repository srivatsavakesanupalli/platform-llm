from .base_trainer import BaseTrainer
from configs.input_config import InputConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import AdamW, DataCollatorWithPadding
from peft import LoraConfig, PeftModel
from accelerate import Accelerator

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from loguru import logger
import evaluate

from datasets import load_dataset


class ClassificationTrainer(BaseTrainer):
    def __init__(
        self, config: InputConfig
    ):
        super().__init__(config)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, "float16"),
            bnb_4bit_use_double_quant=False,
        )
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
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

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.architecture, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
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
                    task_type="SEQ_CLS",
                    target_modules=["q_proj", "k_proj", "v_proj", "score"],
                )
            return PeftModel(self.model, self.peft_config)

    def print_trainable_params(self):
        logger.info(self.model.print_trainable_params())

    def fit_model(self, train_dataloader):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epochs):
            self.model.train()
            logger.info(f"Running Epoch: {epoch+1}")
            total_loss = 0
            for idx, batch in enumerate(train_dataloader):
                if idx % 100 == 0:
                    logger.info(f"\t Running batch: {idx+1}")
                batch_input = self.tokenizer(
                    batch["text"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=200,
                ).to(self.device)
                labels = batch["labels"].unsqueeze(dim=1)
                labels = labels.half().to(self.device)

                with self.accelerator.accumulate(self.model):
                    optimizer.zero_grad()
                    outputs = self.model(**batch_input)
                    logits = outputs.logits
                    loss = criterion(logits, labels.half())
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}] - Loss: {avg_loss}")

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
        self.fit_model(self.trainloader)

        self.save_adaptor()
        logger.info("Saved trained model and tokenizer")

        train_metrics = self.evaluate_fn(self.trainloader)
        eval_metrics = self.evaluate_fn(self.testloader)
        return train_metrics, eval_metrics
