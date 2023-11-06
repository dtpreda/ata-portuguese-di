import sys
import evaluate
import numpy as np

from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer
from transformers import DebertaV2ForSequenceClassification, set_seed, DebertaV2ForMultipleChoice
from huggingface_hub import login

login(token='ADD TOKEN HERE')


arg_task = sys.argv[1]  # 1k-4sent
arg_seed = sys.argv[2]  # 41
arg_context_length = int(sys.argv[3])  # 128
arg_batch_size = int(sys.argv[4])  # 8
arg_lora = sys.argv[5]
set_seed(int(arg_seed))


model = DebertaV2ForSequenceClassification.from_pretrained(arg_lora)
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

metric_f1 = evaluate.load("f1")
metric_acc = evaluate.load("accuracy")


def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        f1 = metric_f1.compute(predictions=predictions, references=labels, average='macro')
        acc = metric_acc.compute(predictions=predictions, references=labels)
        return {"f1": f1['f1'], "accuracy": acc['accuracy']}


args = TrainingArguments(
    do_train=False,
    do_eval=True,
    do_predict=False,
    save_strategy="no",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_eval_batch_size=8,
    num_train_epochs=0,
    weight_decay=0.05,
    metric_for_best_model= "spearmanr" if arg_task == 'stsb' else "f1",
    output_dir="www"
)


trainer = Trainer(
    model,
    args,
    eval_dataset=encoded_dataset["test"] if len(dataset) != 1 else encoded_dataset["train"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

eval = trainer.evaluate()
print(eval)
