from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from constants import OUTPUT_DIR
from evaluation import compute_metrics


def create_training_arguments() -> TrainingArguments:

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,


        num_train_epochs=7,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,


        learning_rate=3e-5,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        lr_scheduler_type="linear",
        warmup_ratio=0.2,


        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",


        fp16=False,
        fp16_full_eval=False,
        dataloader_num_workers=2,
        disable_tqdm=False
    )

    return training_args


def build_trainer(model, tokenizer, tokenized_datasets) -> Trainer:

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args: TrainingArguments = create_training_arguments()

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
