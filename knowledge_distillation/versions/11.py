import functools

import torch.distributed as dist
import torch.sagemaker as tsm

from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy

import argparse
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification
)
from datasets import Dataset
import evaluate

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.sagemaker.delayed_param import DelayedParamIniter
from accelerate import init_empty_weights


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
        "--bf16",
        type=bool,
        default=True,
        help="Whether to use bf16.",
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
    return accuracy.compute(predictions=predictions, references=labels)


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        assert (outputs_student.logits.size() == outputs_teacher.logits.size())

        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1),
        ) * (self.args.temperature ** 2)

        loss = (self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits)
        return (loss, outputs_student) if return_outputs else loss


def training_function(args):
    import smdistributed.dataparallel.torch.torch_smddp
    dist.init_process_group(backend="smddp")

    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()

    tsm.init()

    train_dataset = Dataset.from_file("/opt/ml/input/data/train/data-00000-of-00001.arrow")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
    )

    with init_empty_weights():
        teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model_id, num_labels=2,
                                                                           id2label=id2label, label2id=label2id,
                                                                           cache_dir="/tmp", low_cpu_mem_usage=True,
                                                                           device_map="auto")
    with init_empty_weights():
        student_model = AutoModelForSequenceClassification.from_pretrained(args.student_model_id, num_labels=2,
                                                                           id2label=id2label, label2id=label2id,
                                                                           cache_dir="/tmp", low_cpu_mem_usage=True,
                                                                           device_map="auto")

    delayed_initer_teacher = DelayedParamIniter(teacher_model)
    delayed_initer_student = DelayedParamIniter(student_model)

    torch.cuda.set_device(device)

    with delayed_initer_teacher.validate_params_and_buffers_inited():
        teacher_model = FSDP(
            teacher_model,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            mixed_precision=mixed_precision_policy,
            param_init_fn=delayed_initer_teacher.get_param_init_fn(),
            device_id=torch.cuda.current_device(),
            auto_wrap_policy=functools.partial(size_based_auto_wrap_policy, )
        )

    with delayed_initer_student.validate_params_and_buffers_inited():
        student_model = FSDP(
            student_model,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            mixed_precision=mixed_precision_policy,
            param_init_fn=delayed_initer_student.get_param_init_fn(),
            device_id=torch.cuda.current_device(),
            auto_wrap_policy=functools.partial(size_based_auto_wrap_policy, )
        )

    accuracy = evaluate.load("accuracy")

    training_args = DistillationTrainingArguments(
        output_dir="/opt/ml/model/",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        bf16=args.bf16,
        learning_rate=args.lr,
        evaluation_strategy=args.evaluation_strategy,
        alpha=args.alpha,
        temperature=args.temperature,
        remove_unused_columns=False,
    )

    trainer = DistillationTrainer(
        student_model,
        training_args,
        teacher_model=teacher_model,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()


def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
