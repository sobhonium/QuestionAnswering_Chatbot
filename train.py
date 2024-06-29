import json
import os

from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

import warnings
warnings.filterwarnings("ignore")

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        json_file = json.load(f)
    return json_file    


def createDataset(inp_dict):
    contexts = []
    questions = []
    answers = []

    for article in inp_dict['data']:
 
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qs in paragraph['qas']:
                question = qs['question']
                for answer in qs['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append({'text': answer['text'], 'answer_start': answer['answer_start']})

    # build a dataset and return it as an object from Datset class. 
    return Dataset.from_dict({
    'context': contexts,
    'question': questions,
    'answers': answers
    })

def preprocess_function(examples):
    
    questions = [q.strip() for q in examples["question"]]
    
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=500,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"]
        end_char = answer["answer_start"] + len(answer["text"])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


if __name__ == "__main__":
    dev = read_json_file(file_path='dataset/dev-v1.1.json')
    train = read_json_file(file_path='dataset/train-v1.1.json')

    TRAIN = createDataset(train)
    TEST = createDataset(dev)
    
    TRAIN = TRAIN.train_test_split(test_size=0.2, seed=1)

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
    data_collator = DefaultDataCollator()
    
    train_tokenized = TRAIN.map(preprocess_function, batched=True, remove_columns=TEST.column_names)
    test_tokenized = TEST.map(preprocess_function, batched=True, remove_columns=TEST.column_names)

    training_args = TrainingArguments(
    output_dir="my_model",
    evaluation_strategy="epoch",
    save_strategy='epoch',
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    push_to_hub=False,
    report_to = 'none',
    load_best_model_at_end = True,
    overwrite_output_dir = True,
    metric_for_best_model= "eval_loss",
    greater_is_better= False
    )

    
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized["train"],
    eval_dataset=train_tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    )

    trainer.train()
    
    trainer.save_model('model/final_model')

    trainer.evaluate(eval_dataset=test_tokenized)

