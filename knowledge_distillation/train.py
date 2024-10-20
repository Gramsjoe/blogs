import argparse
import functools

import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.utils
import torch.optim as optim

import torch.sagemaker as tsm

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, \
    DataCollatorForLanguageModeling
from datasets import Dataset
from huggingface_hub import hf_hub_download

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.cuda.amp import GradScaler, autocast

from nltk.translate.bleu_score import sentence_bleu
import smdistributed.dataparallel.torch.torch_smddp
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def parse_arge():
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher_model_id", type=str, default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--student_model_id", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--hf_token", type=str, default="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")
    parser.add_argument("--dist_backend", type=str, default="smddp")

    args = parser.parse_known_args()
    return args


class DistillationLoss(nn.Module):
    def __init__(self, alpha, temperature):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, s_logits, t_logits, target, global_rank):
        s_softmax_log = nn.functional.log_softmax(s_logits / self.temperature, dim=-1)
        t_softmax = nn.functional.softmax(t_logits / self.temperature, dim=-1)

        distillation_loss = self.criterion(s_softmax_log, t_softmax) * (self.temperature ** 2)

        s_logits = s_logits.view(-1, s_logits.size(-1))
        target = target.view(-1)

        student_loss = nn.functional.cross_entropy(s_logits, target)

        loss = (self.alpha * student_loss + (1.0 - self.alpha) * distillation_loss)
        if global_rank == 0:
            print(f'loss: {loss}, student_loss: {student_loss}, distillation_loss: {distillation_loss}')

        return loss


def training_function(args):
    dist.init_process_group(backend=args.dist_backend)
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()

    tsm.init()

    mistral_config = AutoConfig.from_pretrained(args.student_model_id, token=args.hf_token)
    mixtral_config = AutoConfig.from_pretrained(args.teacher_model_id, token=args.hf_token)
    mistral_config.pad_token_id = 0
    mixtral_config.pad_token_id = 0

    if global_rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(args.student_model_id, token=args.hf_token)
        tokenizer.pad_token = tokenizer.unk_token
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model_id,
                                                             cache_dir="/tmp",
                                                             token=args.hf_token,
                                                             config=mixtral_config)
        student_model = AutoModelForCausalLM.from_pretrained(args.student_model_id,
                                                             cache_dir="/tmp",
                                                             token=args.hf_token,
                                                             config=mistral_config)

    else:
        with torch.device("meta"):
            tokenizer = AutoTokenizer.from_pretrained(args.student_model_id, token=args.hf_token)
            tokenizer.pad_token = tokenizer.unk_token
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
            teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model_id,
                                                                 cache_dir="/tmp",
                                                                 token=args.hf_token,
                                                                 config=mixtral_config)
            student_model = AutoModelForCausalLM.from_pretrained(args.student_model_id,
                                                                 cache_dir="/tmp",
                                                                 token=args.hf_token,
                                                                 config=mistral_config)

    train_dataset = Dataset.from_file("/opt/ml/input/data/train/data-00000-of-00001.arrow")

    data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, collate_fn=data_collator)

    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
        cast_forward_inputs=True,
    )

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
    )

    torch.cuda.set_device(device)

    teacher_model = FSDP(
        teacher_model,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
        mixed_precision=mixed_precision_policy,
        sync_module_states=True,
        param_init_fn=(
            lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)) if global_rank != 0 else None,
    )

    student_model = FSDP(
        student_model,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
        mixed_precision=mixed_precision_policy,
        sync_module_states=True,
        param_init_fn=(
            lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)) if global_rank != 0 else None,
    )

    criterion = DistillationLoss(args.alpha, args.temperature)
    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr)
    scaler = GradScaler()

    teacher_model.eval()
    student_model.train()

    for epoch in range(args.num_epochs):

        num_batches = 0

        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            inputs = {k: v.to('cuda') for k, v in batch.items()}

            with autocast():
                with torch.no_grad():
                    teacher_outputs = teacher_model(**inputs)

                student_outputs = student_model(**inputs)

                loss = criterion(student_outputs.logits, teacher_outputs.logits, inputs['labels'], global_rank)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            num_batches += 1

    test_dataset = Dataset.from_file("/opt/ml/input/data/test/data-00000-of-00001.arrow")

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch, shuffle=True,
                                              collate_fn=data_collator)

    student_model.eval()

    all_generated_texts = []
    all_target_texts = []

    for batch in test_loader:
        inputs = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            outputs = student_model(**inputs)
            generated_text = outputs.logits.argmax(dim=-1)
            all_generated_texts.append(generated_text.cpu().numpy())
            all_target_texts.append(inputs['labels'].cpu().numpy())

    bleu_scores = []

    for target, generated in zip(all_target_texts, all_generated_texts):
        reference = [' '.join(map(str, target)).split()]
        candidate = ' '.join(map(str, generated)).split()
        score = sentence_bleu(reference, candidate)
        bleu_scores.append(score)

    average_score = sum(bleu_scores) / len(bleu_scores)

    if global_rank == 0:
        print(f'Average BLEU Score: {average_score}')

    state_dict = student_model.state_dict()

    hf_model = AutoModelForCausalLM.from_config(mistral_config)

    prefix = 'model.'

    new_state_dict = {}

    for key in state_dict.keys():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = state_dict[key]

    hf_model.load_state_dict(new_state_dict, strict=False)

    hf_model.save_pretrained("/opt/ml/model")
    tokenizer.save_pretrained("/opt/ml/model")
    hf_hub_download(repo_id=args.student_model_id, filename='tokenizer.model', local_dir="/opt/ml/model",
                    token=args.hf_token)


def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
