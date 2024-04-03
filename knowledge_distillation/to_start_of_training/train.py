import argparse
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    MistralForCausalLM,
)
from datasets import load_metric, Dataset
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def parse_arge():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--teacher_model_id",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model id to use for teacher.",
    )
    parser.add_argument(
        "--student_model_id",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Model id to use for student.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs to train student for."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=6e-5,
        help="Learning rate to use for training."
    )
    parser.add_argument(
        "--fp16",
        type=bool,
        default=True,
        help="Whether to use fp16.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="How much to soften teacher logits distribution."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Percentage of student loss weighted to hard targets"
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default='epoch',
        help="What training stage to save after.",
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default='epoch',
        help="What training stage to evaluate after.",
    )

    args = parser.parse_known_args()
    return args


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_metric.compute(
        predictions=predictions, references=labels
    )
    return {
        "accuracy": acc["accuracy"],
    }


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True

    def __call__(self, features):
        labels = [feature.pop("answer") for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)

        return batch


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        assert (
                outputs_student.logits.size() == outputs_teacher.logits.size()
        )

        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1),
        ) * (self.args.temperature ** 2)

        loss = (self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits)
        return (loss, outputs_student) if return_outputs else loss


def training_function(args):
    train_dataset = Dataset.from_file("/opt/ml/input/data/train_data/data-00000-of-00001.arrow")
    test_dataset = Dataset.from_file("/opt/ml/input/data/test_data/data-00000-of-00001.arrow")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    teacher_model = MistralForCausalLM.from_pretrained(args.teacher_model_id, cache_dir="/tmp")
    student_model = MistralForCausalLM.from_pretrained(args.student_model_id, cache_dir="/tmp")

    accuracy_metric = load_metric("accuracy")

    training_args = DistillationTrainingArguments(
        output_dir="/opt/ml/model/",
        num_train_epochs=args.num_epochs,
        # auto_find_batch_size=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        fp16=args.fp16,
        learning_rate=args.lr,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        alpha=args.alpha,
        temperature=args.temperature,
    )

    trainer = DistillationTrainer(
        student_model,
        training_args,
        teacher_model=teacher_model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()


def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
