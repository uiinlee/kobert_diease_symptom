#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install torch')
get_ipython().system('pip install accelerate -U')


# In[2]:


import os
import re
import pickle


# In[3]:


from transformers import TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader


# In[4]:


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

sentences_path = r"C:\Users\User\voice sample\labeling data\labeling update\filtered_sentences.pkl"
labels_path = r"C:\Users\User\voice sample\labeling data\labeling update\filtered_labels.pkl"


# In[75]:


class CustomDataset(Dataset):
    def __init__(self, sentences_path, labels_path, tokenizer, max_len):
        with open(sentences_path, 'rb') as f:
            original_sentences = pickle.load(f)
        with open(labels_path, 'rb') as f:
            original_labels = pickle.load(f)
        self.sentences = []
        self.labels = []
        for sentence, label in zip(original_sentences, original_labels):
            tokens = tokenizer.tokenize(sentence)
            if len(tokens) <= max_len:
                self.sentences.append(sentence)
                self.labels.append(label)

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].flatten()
        attn_mask = encoding['attention_mask'].flatten()

        label = [0] + label +[0]

        label += [0] * (self.max_len - len(label))

        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'labels': torch.tensor(label, dtype=torch.long)[:self.max_len]
        }


# In[76]:


MAX_LEN = 50
BATCH_SIZE = 32
train_data = CustomDataset(sentences_path, labels_path, tokenizer, MAX_LEN)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)


# In[5]:


from transformers import BertForTokenClassification, AdamW, BertConfig
model = BertForTokenClassification.from_pretrained("monologg/kobert", num_labels=3)


# In[79]:


training_args = TrainingArguments(
    output_dir='./results',  # output directory for model predictions and checkpoints
    num_train_epochs=15,  # total number of training epochs
    per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
    logging_dir='./logs',  # directory for storing logs
    save_total_limit=3,  # Only the last 3 models will be saved. Older ones are deleted.
)


# In[80]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
) 

trainer.train()


# In[82]:


model.save_pretrained(r"C:\Users\User\Ko Model")


# In[94]:


sentence = "너무 머리가 아파요"
inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2)

for token, label_idx in zip(tokenizer.tokenize(sentence), predictions[0].tolist()):
  label = "증상" if label_idx == 1 else "기타"
  print(f"{token} : {label}")


# In[100]:


get_ipython().system('pip install scikit-learn')


# In[107]:


from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for l in label if l != -100]
        for label in labels
    ]

    accuracy = accuracy_score(true_labels, true_predictions)
    return {
        'accuracy_percentage': accuracy * 100  # 정확도를 백분율로 변환
    }


# In[7]:


from transformers import BertForTokenClassification
model = BertForTokenClassification.from_pretrained(r"C:\Users\User\Ko Model")


# In[12]:


import pickle
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

# 피클 파일 경로
filtered_sentences_dir = r"C:\Users\User\voice sample\test\labeling\filtered\filtered_sentences.pkl"
filtered_labels_dir = r"C:\Users\User\voice sample\test\labeling\filtered\filtered_labels.pkl"

# 피클 파일 로드
with open(filtered_sentences_dir, 'rb') as f:
    test_sentences = pickle.load(f)
with open(filtered_labels_dir, 'rb') as f:
    test_labels = pickle.load(f)

# 모델 초기화 (이미 학습된 모델을 불러와야 함)
model = BertForTokenClassification.from_pretrained(r"C:\Users\User\Ko Model")
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

predicted_labels = []

# 예측
for sentence in test_sentences:
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=2).tolist()[0]
    predicted_labels.append(predicted_label)

# 다중 레이블 정확도 계산
mlb = MultiLabelBinarizer()
binarized_test_labels = mlb.fit_transform(test_labels)
binarized_predicted_labels = mlb.transform(predicted_labels)
multi_label_accuracy = accuracy_score(binarized_test_labels, binarized_predicted_labels)

print(f"다중 레이블 정확도: {multi_label_accuracy * 100:.2f}%")


# In[13]:


example_filtered_sentences_dir = r"C:\Users\User\voice sample\test\labeling\filtered\filtered_sentences.pkl"
with open(example_filtered_sentences_dir, 'rb') as f:
    example_test_sentences = pickle.load(f)
print(f"피클 파일에 있는 문장 개수: {len(example_test_sentences)}")

