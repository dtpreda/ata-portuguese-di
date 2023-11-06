import sys
import evaluate
import numpy as np

from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer
from transformers import DebertaV2ForSequenceClassification, set_seed, DebertaV2ForMultipleChoice
from huggingface_hub import login

login(token='ADD_YOURS')


arg_task = sys.argv[1]  # 1k-4sent
arg_variant = sys.argv[2]  # pt, br
arg_seed = sys.argv[3]  # 41
arg_context_length = int(sys.argv[4])  # 128
arg_batch_size = int(sys.argv[5])  # 8
arg_model_version = sys.argv[6]  # 900m or 1.5b
set_seed(int(arg_seed))

model_name = f"PORTULAN/albertina-{arg_model_version}-portuguese-pt{arg_variant}-encoder"


model = DebertaV2ForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge")


dataset = load_dataset(f'data/tedx_parallel_pt_br/{arg_task}')
remove_columns = [col for col in dataset['train'].column_names if col not in ["text", "label_bool"]]
dataset = dataset.remove_columns(remove_columns)
dataset = dataset.rename_column("label_bool", "label")

print(dataset)
print(dataset["train"])
print(dataset["train"][0])


def preprocess_data(examples):
    encoding = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=arg_context_length)
    return encoding


encoded_dataset = dataset.map(preprocess_data, batched=True)
encoded_dataset.set_format("torch")


config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(model, config)

batch_size = arg_batch_size

args = TrainingArguments(
    f"models/lora-albertina-{arg_task}-{arg_variant}-{arg_seed}-{arg_context_length}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.05,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

metric_f1 = evaluate.load("f1")
metric_acc = evaluate.load("accuracy")


def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        f1 = metric_f1.compute(predictions=predictions, references=labels)
        acc = metric_acc.compute(predictions=predictions, references=labels)
        return {"f1": f1['f1'], "accuracy": acc['accuracy']}

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

eval = trainer.evaluate(encoded_dataset["test"])
print(arg_task)
print(arg_variant)
print(eval)
print(arg_model_version)
trainer.save_model(f"models/lora-albertina-{arg_model_version}-ptvsbr-{arg_task}-{arg_variant}-{arg_seed}-{arg_context_length}/final")

