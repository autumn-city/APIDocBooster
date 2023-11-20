from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import sklearn
import os
import data_preprocess
import sklearn
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
import torch
import shutil
import pickle

class CustomTrainer(Trainer):
    # context_pair = load_dataset('../../data/classification_data/single_input')

    def compute_loss(self, model, inputs, return_outputs=False):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'


        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # print(self.model.config.num_labels.device)
        num_labels = torch.tensor(self.model.config.num_labels, dtype=torch.int8)
        # print(labels.device);exit()

        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_wts)).to(device)

        logits.cuda()
        labels.cuda()
        self.model.cuda()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1)).to(device)

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    f1 = evaluate.load("f1")
    predictions, labels = eval_pred
    # print(predictions[0])
    predictions = np.argmax(predictions, axis=-1)
    return f1.compute(predictions=predictions, references=labels, average='micro')

def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(examples["text1"],examples["text2"], truncation=True)

def preprocess_function_single(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(examples["text"], truncation=True)

def context_trainer(addr):

    context_pair = load_dataset('../../data/context/')

    global class_wts
    class_wts = compute_class_weight(class_weight = 'balanced', classes= np.unique(context_pair['train']['label']), y=context_pair['train']['label']).astype(np.float32)


    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized_dataset = context_pair.map(preprocess_function_single, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['text','text2'])   
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # label-map
    id2label = {0: "NEGATIVE", 1: "function", 2: "parameter", 3: "others"}
    label2id = {"NEGATIVE": 0, "function": 1,"parameter": 1,"others": 3}

    # train
    model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
).to('cuda')
    training_args = TrainingArguments(
    output_dir=addr,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    return tokenized_dataset


def inference_single():

    context_pair = load_dataset('../../data/context/')

    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    id2label = {0: "NEGATIVE", 1: "function", 2: "parameter", 3: "others"}
    label2id = {"NEGATIVE": 0, "function": 1,"parameter": 1,"others": 3}
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_dataset = context_pair.map(preprocess_function_single, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['text','text2'])   
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

     # tokenized_dataset = context_pair.map(preprocess_function, batched=True)
    print(tokenized_dataset)

    model = AutoModelForSequenceClassification.from_pretrained("model/checkpoint-546", num_labels=2 id2label=id2label, label2id=label2id)

    training_args = TrainingArguments(
    output_dir="contextmodel",
    do_train = False,
    do_predict = True,
    dataloader_drop_last = False    
)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    metric = trainer.predict(tokenized_dataset["test"])
    print(metric[0])
    # metric = trainer.evaluate()
    predictions = metric.predictions
    print(predictions[:10])
    print(predictions[0])
    print(torch.softmax(torch.Tensor(predictions[0]),0))
    print(np.argmax(predictions[:10], axis=-1))
    print(metric.label_ids[:10])
    # print(np.argmax(predictions, axis=-1))
    # print(metric.label_ids)
    print(sklearn.metrics.classification_report(np.argmax(predictions, axis=-1),metric.label_ids))

    return predictions


if __name__ == '__main__':
    context_trainer('contextmodel')
