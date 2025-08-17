# scripts/train_nli.py
# BharatWitness NLI model training and calibration script

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import pickle
from tqdm import tqdm

from utils.logging_utils import setup_logging
from utils.seed_utils import set_deterministic_seed
from pipeline.claim_verification import ClaimExtractor



class NLIDataset(Dataset):
    def __init__(self, premise_list: List[str], hypothesis_list: List[str], labels: List[int], tokenizer,
                 max_length: int = 512):
        self.premise_list = premise_list
        self.hypothesis_list = hypothesis_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.premise_list)

    def __getitem__(self, idx):
        premise = str(self.premise_list[idx])
        hypothesis = str(self.hypothesis_list[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            hypothesis,
            premise,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class NLITrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.nli_config = self.config["nli"]
        self.model_name = self.nli_config["model"]
        self.batch_size = self.nli_config["batch_size"]

        self.models_root = Path(self.config["paths"]["models_root"])
        self.output_dir = self.models_root / "nli"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("bharatwitness.nli_trainer")
        set_deterministic_seed(self.config["system"]["seed"])

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.label_mapping = {
            'entailment': 0,
            'neutral': 1,
            'contradiction': 2,
            'supported': 0,
            'uncertain': 1,
            'refuted': 2
        }

    def prepare_training_data(self, data_file: Path) -> Tuple[List[str], List[str], List[int]]:
        self.logger.info(f"Loading training data from {data_file}")

        if data_file.suffix == '.json':
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_file.suffix == '.jsonl':
            data = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            raise ValueError("Data file must be JSON or JSONL format")

        premises = []
        hypotheses = []
        labels = []

        for item in data:
            premise = item.get('premise', item.get('evidence', ''))
            hypothesis = item.get('hypothesis', item.get('claim', ''))
            label = item.get('label', item.get('gold_label', ''))

            if premise and hypothesis and label in self.label_mapping:
                premises.append(premise)
                hypotheses.append(hypothesis)
                labels.append(self.label_mapping[label])

        self.logger.info(f"Prepared {len(premises)} training examples")
        return premises, hypotheses, labels

    def create_synthetic_legal_data(self, corpus_file: Path, num_samples: int = 1000) -> Tuple[
        List[str], List[str], List[int]]:
        self.logger.info("Creating synthetic legal NLI data")

        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)

        claim_extractor = ClaimExtractor()

        premises = []
        hypotheses = []
        labels = []

        chunk_texts = [chunk_data['text'] for chunk_data in corpus_data.values()]

        for i, text in enumerate(tqdm(chunk_texts[:num_samples // 3], desc="Generating synthetic data")):
            claims = claim_extractor.extract_claims(text)

            for claim in claims[:3]:
                claim_text = claim['text']

                premises.append(text)
                hypotheses.append(claim_text)
                labels.append(0)

                negated_claim = self._create_negated_claim(claim_text)
                if negated_claim:
                    premises.append(text)
                    hypotheses.append(negated_claim)
                    labels.append(2)

                modified_claim = self._create_modified_claim(claim_text)
                if modified_claim:
                    premises.append(text)
                    hypotheses.append(modified_claim)
                    labels.append(1)

        self.logger.info(f"Generated {len(premises)} synthetic examples")
        return premises, hypotheses, labels

    def _create_negated_claim(self, claim: str) -> str:
        if 'shall' in claim.lower():
            return claim.lower().replace('shall', 'shall not')
        elif 'must' in claim.lower():
            return claim.lower().replace('must', 'must not')
        elif 'is' in claim.lower():
            return claim.lower().replace('is', 'is not')
        elif 'are' in claim.lower():
            return claim.lower().replace('are', 'are not')
        else:
            return f"It is not true that {claim.lower()}"

    def _create_modified_claim(self, claim: str) -> str:
        import random

        modifications = [
            lambda x: x.replace('10%', '15%') if '10%' in x else x,
            lambda x: x.replace('shall', 'may') if 'shall' in x.lower() else x,
            lambda x: x.replace('must', 'should') if 'must' in x.lower() else x,
            lambda x: x.replace('all', 'some') if 'all' in x.lower() else x,
            lambda x: x.replace('every', 'any') if 'every' in x.lower() else x
        ]

        modification = random.choice(modifications)
        return modification(claim)

    def train_model(self, premises: List[str], hypotheses: List[str], labels: List[int], epochs: int = 3,
                    eval_split: float = 0.2) -> str:
        self.logger.info("Starting NLI model training")

        split_idx = int(len(premises) * (1 - eval_split))

        train_premises = premises[:split_idx]
        train_hypotheses = hypotheses[:split_idx]
        train_labels = labels[:split_idx]

        eval_premises = premises[split_idx:]
        eval_hypotheses = hypotheses[split_idx:]
        eval_labels = labels[split_idx:]

        train_dataset = NLIDataset(train_premises, train_hypotheses, train_labels, self.tokenizer)
        eval_dataset = NLIDataset(eval_premises, eval_hypotheses, eval_labels, self.tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            id2label={0: 'entailment', 1: 'neutral', 2: 'contradiction'},
            label2id={'entailment': 0, 'neutral': 1, 'contradiction': 2}
        )

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics
        )

        trainer.train()

        model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        self.logger.info(f"Model saved to {self.output_dir}")
        return str(self.output_dir)

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def calibrate_model(self, premises: List[str], hypotheses: List[str], labels: List[int]) -> str:
        self.logger.info("Calibrating model confidence scores")

        model = AutoModelForSequenceClassification.from_pretrained(str(self.output_dir))
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()

        probabilities = []
        true_labels = []

        dataset = NLIDataset(premises, hypotheses, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting predictions for calibration"):
                if torch.cuda.is_available():
                    batch = {k: v.cuda() for k, v in batch.items()}

                outputs = model(**batch)
                logits = outputs.logits

                batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()
                batch_labels = batch['labels'].cpu().numpy()

                probabilities.extend(batch_probs)
                true_labels.extend(batch_labels)

        probabilities = np.array(probabilities)
        true_labels = np.array(true_labels)

        calibrator = CalibratedClassifierCV(cv=3, method='isotonic')
        calibrator.fit(probabilities, true_labels)

        calibrator_path = self.output_dir / "calibrator.pkl"
        with open(calibrator_path, 'wb') as f:
            pickle.dump(calibrator, f)

        self.logger.info(f"Calibrator saved to {calibrator_path}")
        return str(calibrator_path)

    def evaluate_model(self, test_premises: List[str], test_hypotheses: List[str], test_labels: List[int]) -> Dict[
        str, Any]:
        self.logger.info("Evaluating trained model")

        model = AutoModelForSequenceClassification.from_pretrained(str(self.output_dir))
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()

        test_dataset = NLIDataset(test_premises, test_hypotheses, test_labels, self.tokenizer)
        dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if torch.cuda.is_available():
                    batch = {k: v.cuda() for k, v in batch.items()}

                outputs = model(**batch)
                logits = outputs.logits

                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                labels = batch['labels'].cpu().numpy()

                all_predictions.extend(predictions)
                all_labels.extend(labels)
                all_probabilities.extend(probabilities)

        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

        conf_matrix = confusion_matrix(all_labels, all_predictions)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'num_samples': len(all_labels)
        }


def main():
    parser = argparse.ArgumentParser(description="Train BharatWitness NLI verification model")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--data", help="Training data file (JSON/JSONL)")
    parser.add_argument("--corpus", help="Corpus file for synthetic data generation")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--synthetic-samples", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate confidence scores")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing model")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode")

    args = parser.parse_args()

    logger = setup_logging(args.config)
    trainer = NLITrainer(args.config)

    if args.eval_only:
        if not args.data:
            logger.error("Test data file required for evaluation")
            return 1

        premises, hypotheses, labels = trainer.prepare_training_data(Path(args.data))
        results = trainer.evaluate_model(premises, hypotheses, labels)

        logger.info("Evaluation Results:")
        for metric, value in results.items():
            if metric != 'confusion_matrix':
                logger.info(f"{metric}: {value:.4f}")

        return 0

    if args.data:
        premises, hypotheses, labels = trainer.prepare_training_data(Path(args.data))
    elif args.corpus:
        premises, hypotheses, labels = trainer.create_synthetic_legal_data(
            Path(args.corpus),
            args.synthetic_samples
        )
    else:
        logger.error("Either --data or --corpus must be provided")
        return 1

    model_path = trainer.train_model(premises, hypotheses, labels, args.epochs)
    logger.info(f"Model training completed: {model_path}")

    if args.calibrate:
        calibrator_path = trainer.calibrate_model(premises, hypotheses, labels)
        logger.info(f"Model calibration completed: {calibrator_path}")

    test_split = int(len(premises) * 0.8)
    test_premises = premises[test_split:]
    test_hypotheses = hypotheses[test_split:]
    test_labels = labels[test_split:]

    if test_premises:
        results = trainer.evaluate_model(test_premises, test_hypotheses, test_labels)
        logger.info("Final Test Results:")
        for metric, value in results.items():
            if metric != 'confusion_matrix':
                logger.info(f"{metric}: {value:.4f}")

    return 0


if __name__ == "__main__":
    exit(main())
