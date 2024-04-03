import argparse
import os

from datasets import load_dataset
from setfit import sample_dataset
from setfit import SetFitModel, TrainingArguments, Trainer
from setfit import DistillationTrainer
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


def parse_arge():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--teacher_model_id",
        type=str,
        default="sentence-transformers/paraphrase-mpnet-base-v2",
        help="Model id to use for teacher.",
    )
    parser.add_argument(
        "--student_model_id",
        type=str,
        default="sentence-transformers/paraphrase-MiniLM-L3-v2",
        help="Model id to use for student.",
    )
    parser.add_argument(
        "--student_num_epochs_labelled",
        type=int,
        default=5,
        help="Number of epochs to train student for with labelled data."
    )
    parser.add_argument(
        "--student_batch_size_labelled",
        type=int,
        default=64,
        help="Batch size to use for training student with labelled data."
    )
    parser.add_argument(
        "--teacher_num_epochs",
        type=int,
        default=2,
        help="Number of epochs to train teacher for."
    )
    parser.add_argument(
        "--teacher_batch_size",
        type=int,
        default=16,
        help="Batch size to use for training teacher."
    )
    parser.add_argument(
        "--distillation_batch_size",
        type=int,
        default=16,
        help="Batch size to use for distillation training."
    )
    parser.add_argument(
        "--distillation_trainer_max_steps",
        type=int,
        default=500,
        help="Max steps to use for distillation training."
    )
    parser.add_argument(
        "--teacher_train_data_num_samples",
        type=int,
        default=16,
        help="Number of labelled samples to use when training teacher."
    )
    parser.add_argument(
        "--student_train_data_num_samples",
        type=int,
        default=500,
        help="Number of unlabelled samples to use when training student."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ag_news",
        help="Dataset to use for training and evaluation"
    )
    parser.add_argument(
        "--dataset_label_column",
        type=str,
        default="label",
        help="Column of training dataset with correct label"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="/opt/ml/checkpoints/",
        help="folder to save model checkpoints in"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate to use for training."
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--fp16",
        type=bool,
        default=True,
        help="Whether to use fp16.",
    )
    parser.add_argument(
        "--save_embeddings",
        type=bool,
        default=True,
        help="whether to save embeddings for merged model.",
    )
    parser.add_argument(
        "--training_output_dir",
        type=str,
        default="/tmp1",
        help="Where to save training job model.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data-00000-of-00001.arrow",
        help="Dataset name.",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default='epoch',
        help="What training stage to save after.",
    )

    args = parser.parse_known_args()
    return args


def custom_metric_fn(y_pred, y_test):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')  # change averaging strategy as needed
    precision = precision_score(y_test, y_pred, average='macro')  # change averaging strategy as needed
    recall = recall_score(y_test, y_pred, average='macro')  # change averaging strategy as needed

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def training_function(args):
    metrics_file_path = "/opt/ml/output/metrics/"
    dataset = load_dataset(args.dataset)

    train_dataset = sample_dataset(dataset["train"], label_column=args.dataset_label_column,
                                   num_samples=args.teacher_train_data_num_samples)
    eval_dataset = dataset["test"]

    student_model = SetFitModel.from_pretrained(args.student_model_id)

    student_args = TrainingArguments(
        batch_size=args.student_batch_size_labelled,
        num_epochs=args.student_num_epochs_labelled,
    )

    trainer = Trainer(
        model=student_model,
        args=student_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    student_metrics = trainer.evaluate()

    unlabeled_train_dataset = dataset["train"].shuffle(seed=0).select(range(args.student_train_data_num_samples))
    unlabeled_train_dataset = unlabeled_train_dataset.remove_columns(args.dataset_label_column)

    teacher_model = SetFitModel.from_pretrained(args.teacher_model_id)

    teacher_args = TrainingArguments(
        batch_size=args.teacher_batch_size,
        num_epochs=args.teacher_num_epochs,
    )

    teacher_trainer = Trainer(
        model=teacher_model,
        args=teacher_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    teacher_trainer.train()
    teacher_metrics = teacher_trainer.evaluate()

    distillation_args = TrainingArguments(
        batch_size=args.distillation_batch_size,
        max_steps=args.distillation_trainer_max_steps,
        output_dir=args.checkpoints,
        evaluation_strategy="epoch",
        logging_dir=f"{args.checkpoints}/logs",
    )

    distillation_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        args=distillation_args,
        train_dataset=unlabeled_train_dataset,
        eval_dataset=eval_dataset,
        metric=custom_metric_fn,
    )

    distillation_trainer.train()
    distillation_metrics = distillation_trainer.evaluate()
    with open(os.path.join(args.checkpoints, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(distillation_metrics.items()):
            writer.write(f"{key} = {value}\n")

    distillation_trainer.model.save_pretrained("/opt/ml/model/")


def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
