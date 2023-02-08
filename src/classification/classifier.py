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
    # apisum = load_dataset('../../data/classification_data/single_input')

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

def main_single_input(addr):

    apisum = load_dataset('../../data/classification_data/')

    global class_wts
    class_wts = compute_class_weight(class_weight = 'balanced', classes= np.unique(apisum['train']['label']), y=apisum['train']['label']).astype(np.float32)


    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized_dataset = apisum.map(preprocess_function_single, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['text','text2'])   
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # label-map
    id2label = {0: "NEGATIVE", 1: "function", 2: "parameter", 3: "others"}
    label2id = {"NEGATIVE": 0, "function": 1,"parameter": 1,"others": 3}

    # train
    model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=4, id2label=id2label, label2id=label2id
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

    # apisum = load_dataset('../../data/classification_data/input_with_query')
    apisum = load_dataset('../../data/classification_data/API_split')

    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    id2label = {0: "NEGATIVE", 1: "function", 2: "parameter", 3: "others"}
    label2id = {"NEGATIVE": 0, "function": 1,"parameter": 1,"others": 3}
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_dataset = apisum.map(preprocess_function_single, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['text','text2'])   
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

     # tokenized_dataset = apisum.map(preprocess_function, batched=True)
    print(tokenized_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(
    "model/checkpoint-546", num_labels=4, id2label=id2label, label2id=label2id)

    logits = []
    for inputs in tokenized_dataset['test']:
        with torch.no_grad():
            logits.append(model(**inputs).logits)

    return logits 



def inference_single_input(addr1, input):

    # apisum = load_dataset('../../data/classification_data/input_with_query')
    apisum = load_dataset('../../data/classification_data/')

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    id2label = {0: "NEGATIVE", 1: "function", 2: "parameter", 3: "others"}
    label2id = {"NEGATIVE": 0, "function": 1,"parameter": 1,"others": 3}
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
    addr1+"/checkpoint-206", num_labels=4, id2label=id2label, label2id=label2id)

    # obtain the logits from the model
    with torch.no_grad():
        logits = model(**inputs).logits

    return logits 

def logits_adjust(logits_origin,tokenized_dataset):
    # adjust the logits to the final result
    
    # obtain the context-dependent data
    context_record = pickle.load(open('../../data/classification_data/context_record.pkl','rb'))
    overall_logits = []
    for index, item in enumerate(context_record):
        logits_update = []
        if item['context_pre']>=0.5 and item['context_next']>=0.5:
            logits_context_pre = inference_single_input('model/', tokenized_dataset['test'][index-1]+tokenized_dataset['test'][index])
            logits_context_after = inference_single_input('model/', tokenized_dataset['test'][index]+tokenized_dataset['test'][index+1])
            for i in range(len(logits_origin)):
                logits_update.append((((1-item['context_pre'])*logits_origin[i]+item['context_pre']*logits_context_pre[i])+((1-item['context_pre'])*logits_origin[i]+item['context_pre']*logits_context_after[i]))/2)
        if item['context_pre']>=0.5 and item['context_next']<0.5:
            logits_context_pre = inference_single_input('model/', tokenized_dataset['test'][index-1]+tokenized_dataset['test'][index])
            for i in range(len(logits_origin)):
                logits_update.append(((1-item['context_pre'])*logits_origin[i]+item['context_pre']*logits_context_pre[i]))
        if item['context_pre']<0.5 and item['context_next']>=0.5:
            logits_context_pre = inference_single_input('model/', tokenized_dataset['test'][index-1]+tokenized_dataset['test'][index])
            for i in range(len(logits_origin)):
                logits_update.append(((1-item['context_pre'])*logits_origin[i]+item['context_pre']*logits_context_pre[i]))
        else:
            logits_update = logits_origin
        overall_logits.append(logits_update)
    
    return overall_logits



def reduce_redundancy(addr):
    root = '/workspace/src/classification/'+addr
    for item in os.listdir(root):
        print(item)
        for file in os.listdir(os.path.join(root, item)):
            str1 = []
            print(file)
            with open(os.path.join(root, item,file),'r') as f:
                for line in f.readlines():
                    str1.append(line.replace('\n',''))
            print(len(str1))
            final = list(set(str1))
            with open(os.path.join(root, item,file),'w') as f:
                for sent in final:
                    f.writelines(sent)
                    f.writelines('\n')

if __name__ == '__main__':

    # training stage
    # train the model after the first fine tune
    tokenized_dataset = main_single_input('model')
    # reduce_redundancy('result')

    # inference stage
    # calculate the initial logits
    logits_origin = inference_single('model')
    # update the logits
    logits_update = logits_adjust(logits_origin,tokenized_dataset)
