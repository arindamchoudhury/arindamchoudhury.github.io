## Steps to Finetune BERT

- First, we need to import all the libraries that we need. We will use the **datasets** library to load data as well as functions to compute metrics and from HuggingFace's **transformers** library, we will import tokenizers, trainers, and models for sentence classification.
```python
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
```
- Next, we will define some functions to compute our metrics and tokenize our sentences
```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
```

- Now we can load and preprocess our dataset. Remember that we will be using the **datasets** library to [load data](https://huggingface.co/docs/datasets/v2.14.0/loading). Datasets have many inbuilt datasets available and you can find a list of them [here](https://huggingface.co/datasets).

- The tokenizer we select needs to be the same as the model we are using. There are many pre-trained models available in **transformers** and you can find a list of them [here](https://huggingface.co/transformers/pretrained_models.html). In the code below, you can see that I am using the **bert-base-cased** model. Once we have selected the model, we need to tokenize our dataset. I have also added code to use a small subset of the data to make training faster. However, you may choose to use the whole dataset by uncommenting the last two lines.

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
# full_train_dataset = tokenized_datasets["train"]
# full_eval_dataset = tokenized_datasets["test"]
```

- Now that we have written our data preprocessing code, we can download our model and start to train it. We will use the **AutoModelForSequenceClassification** API to fetch the pre-trained **bert-base-cased** model. We also need to specify the number of classes in our data.


Finally, we can train and evaluate the model using a **Trainer** object.
```python
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=<your labels>)

metric = load_metric("accuracy")

training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.evaluate()
```
