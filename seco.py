import numpy as np
import pandas as pd
import pycaret
import transformers
from transformers import AutoModel, BertTokenizerFast
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

true_data = pd.read_csv('dataset/True.csv')
fake_data = pd.read_csv('dataset/Fake.csv')

true_data['Target'] = ['True']*len(true_data)
fake_data['Target'] = ['Fake']*len(fake_data)

data = pd.concat([true_data, fake_data], ignore_index=True).sample(frac=1).reset_index()

data['label'] = pd.get_dummies(data.Target)['Fake']

label_size = [data['label'].sum(),len(data['label'])-data['label'].sum()]

train_text, temp_text, train_labels, temp_labels = train_test_split(data['text'], data['label'],
                                                                    random_state=2018,
                                                                    test_size=0.3,
                                                                    stratify=data['Target'])

val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=2018,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

bert = AutoModel.from_pretrained('bert')
tokenizer = BertTokenizerFast.from_pretrained('bert')

seq_len = [len(title.split()) for title in train_text]
pd.Series(seq_len).hist(bins = 40,color='firebrick')
plt.xlabel('Number of Words')
plt.ylabel('Number of texts')

MAX_LENGTH = 32

tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = MAX_LENGTH,
    pad_to_max_length=True,
    truncation=True
)

tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = MAX_LENGTH,
    pad_to_max_length=True,
    truncation=True
)

tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = MAX_LENGTH,
    pad_to_max_length=True,
    truncation=True
)

train_seq = torch.tensor(tokens_train['input_ids']).to(device)
train_mask = torch.tensor(tokens_train['attention_mask']).to(device)
train_y = torch.tensor(train_labels.tolist()).to(device)
val_seq = torch.tensor(tokens_val['input_ids']).to(device)
val_mask = torch.tensor(tokens_val['attention_mask']).to(device)
val_y = torch.tensor(val_labels.tolist()).to(device)
test_seq = torch.tensor(tokens_test['input_ids']).to(device)
test_mask = torch.tensor(tokens_test['attention_mask']).to(device)
test_y = torch.tensor(test_labels.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
batch_size = 32                                               
train_data = TensorDataset(train_seq, train_mask, train_y)    
train_sampler = RandomSampler(train_data)                     
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)            
val_sampler = SequentialSampler(val_data)                     
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

for name, param in bert.named_parameters():
    if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

class BERT_Arch(nn.Module):
    def __init__(self, bert):
      super(BERT_Arch, self).__init__()
      self.bert = bert
      self.dropout = nn.Dropout(0.1)            
      self.relu =  nn.ReLU()                    
      self.fc1 = nn.Linear(768,512)             
      self.fc2 = nn.Linear(512,2)               
      self.softmax = nn.LogSoftmax(dim=1)       
    def forward(self, sent_id, mask):           
      cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']

      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)                           
      x = self.softmax(x)                       
      return x
model = BERT_Arch(bert)

model.to(device)

optimizer = optim.AdamW(model.parameters(), lr = 1e-5)

cross_entropy  = nn.NLLLoss()

epochs = 10

def train():
    model.train()
    total_loss, total_accuracy = 0, 0

    for step, batch in enumerate(train_dataloader):  
        if step % 50 == 0 and not step == 0:  
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        batch = [r.to(device) for r in batch]  
        sent_id, mask, labels = batch
        model.zero_grad()  
        preds = model(sent_id, mask)  
        loss = cross_entropy(preds, labels.long())  
        total_loss = total_loss + loss.item()  
        loss.backward()  
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       1.0)  
        optimizer.step()  
        preds = preds.detach().cpu().numpy()  

    avg_loss = total_loss / len(train_dataloader)  
    return avg_loss  

def evaluate():
    print("\nEvaluating...")
    model.eval()  
    total_loss, total_accuracy = 0, 0
    for step, batch in enumerate(val_dataloader):  
        if step % 50 == 0 and not step == 0:  
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        batch = [t.to(device) for t in batch]  
        sent_id, mask, labels = batch
        with torch.no_grad():  
            preds = model(sent_id, mask)  
            loss = cross_entropy(preds, labels.long())  
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
    avg_loss = total_loss / len(val_dataloader)  
    return avg_loss

best_valid_loss = float('inf')
train_losses = []  
valid_losses = []

for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    train_loss = train()  
    valid_loss = evaluate()  
    if valid_loss < best_valid_loss:  
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'c2_new_model_weights.pt')
    train_losses.append(train_loss)  
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

path = 'c2_new_model_weights.pt'
model.load_state_dict(torch.load(path))

with torch.no_grad():
  preds = model(test_seq, test_mask)
  preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds))