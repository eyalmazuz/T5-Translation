import argparse 
from datetime import datetime
from functools import partial

from datasets import load_dataset
import evaluate

import numpy as np

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Config, T5TokenizerFast
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, SchedulerType
from torch.distributed.elastic.multiprocessing.errors import record

ter = evaluate.load("ter")
meteor = evaluate.load('meteor')
chrf = evaluate.load('chrf')
sacrebleu = evaluate.load("sacrebleu")
# bleu = evaluate.load("bleu")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--validation_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--model', type=str, required=True)

    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--logging_steps', type=int, default=5000)
    parser.add_argument('--save_steps', type=int, default=25000)
    parser.add_argument('--local_rank', type=int)

    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    return parser.parse_args()


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]
        # preds = np.argmax(preds[0], -1)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    ter_score = ter.compute(predictions=decoded_preds, references=decoded_labels)
    ter_score = {f'ter_{k}': v for k, v in ter_score.items()}

    meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)

    chrf_score = chrf.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)
    chrf_score = {f'chrf_{k}': v for k, v in chrf_score.items()}
    
    sacrebleu_score = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    sacrebleu_score = {f'scarebleu_{k}': v for k, v in sacrebleu_score.items()}

    # bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, tokenizer=tokenizer)
    # bleu_score = {f'bleu_{k}': v for k, v in bleu_score.items()}

    return {**ter_score, **meteor_score, **chrf_score, **sacrebleu_score,} # **bleu_score}


def preprocess_function(examples, tokenizer, max_length=128):
    inputs = [ex["ar"] for ex in examples["translation"]]
    targets = [ex["he"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs

@record
def main():
    args = parse_args()

    print('Loading data')

    data_files = {'train': args.dataset_path}
    if args.validation_path:
        data_files['validation'] = args.validation_path

    if args.test_path:
        data_files['test'] = args.test_path

    datasets = load_dataset("json", data_files=data_files, field="data")
    datasets["train"] = datasets["train"].shuffle(seed=42)

    if args.validation_path is None:
        print('Splitting data')
        datasets = datasets["train"].train_test_split(train_size=0.99, seed=20)
        datasets["validation"] = datasets.pop("test")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print('Map data')
    tokenized_datasets = datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=datasets["train"].column_names,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': args.max_length},

    )

    config = T5Config.from_pretrained(f'{args.model}')
    config.decoder_start_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.vocab_size = tokenizer.vocab_size

    model = T5ForConditionalGeneration(config)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./results/{args.model} Weight Decay {args.weight_decay} LR {args.learning_rate} Max Length {args.max_length} {str(datetime.now())}",
        learning_rate=args.learning_rate,
        warmup_steps=4000,
        lr_scheduler_type=SchedulerType.COSINE,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        # eval_accumulation_steps=32,
        fp16=True,
        fsdp=["full_shard"],
        fsdp_transformer_layer_cls_to_wrap="T5Block",
        weight_decay=args.weight_decay,
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to="wandb",
        ddp_find_unused_parameters=False,
        # debug='underflow_overflow',
        run_name=f"{args.model} Weight Decay {args.weight_decay} LR {args.learning_rate} Max Length {args.max_length}",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        predict_with_generate=True,
        generation_num_beams=5,
        gradient_accumulation_steps=16,
        generation_max_length=args.max_length,
        sortish_sampler=True,
        save_total_limit=10,
        num_train_epochs=args.epochs,
        # push_to_hub=True
    )


    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"] if args.test_path is None else {'validation': tokenized_datasets["validation"], 'test': tokenized_datasets["test"]},
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
    )

    print('Training')
    trainer.train()


if __name__ == "__main__": 
     main() 
