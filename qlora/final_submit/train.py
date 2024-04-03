import argparse

import torch
import transformers
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def parse_arge():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="tiiuae/falcon-7b",
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train for."
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
        default=True,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="r value to use for Lora model."
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="alpha value to use for Lora model."
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="dropout to use for Lora model.",
    )
    parser.add_argument(
        "--save_embeddings",
        type=bool,
        default=True,
        help="whether to save embeddings for merged model.",
    )
    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=True,
        help="Whether to merge LoRA weights with base model.",
    )
    parser.add_argument(
        "--training_output_dir",
        type=str,
        default="/tmp",
        help="Where to save training job model.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data-00000-of-00001.arrow",
        help="Dataset name.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=4,
        help="Max number of saves during training process.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Number of steps to wait before logging training status.",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default='epoch',
        help="What training stage to save after.",
    )

    args = parser.parse_known_args()
    return args


def training_function(args):
    train_dataset = Dataset.from_file("/opt/ml/input/data/s3_data/" + args.train_file)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = prepare_model_for_kbit_training(base_model)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    peft_model = get_peft_model(base_model, config)

    training_args = transformers.TrainingArguments(
        auto_find_batch_size=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=args.bf16,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        output_dir="/opt/ml/output/data",
        save_strategy=args.save_strategy,
    )

    trainer = transformers.Trainer(
        model=peft_model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    tokenizer.save_pretrained("/opt/ml/model/")

    if args.merge_weights:
        trainer.model.save_pretrained(args.training_output_dir)
        del base_model
        del trainer
        torch.cuda.empty_cache()
        original_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            trust_remote_code=True,
        )

        model_to_merge = PeftModel.from_pretrained(original_model, args.training_output_dir)

        merged_model = model_to_merge.merge_and_unload()
        merged_model.save_pretrained("/opt/ml/model/", save_embedding_layers=args.save_embeddings)
    else:
        trainer.model.save_pretrained("/opt/ml/model/", save_embedding_layers=args.save_embeddings)


def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
