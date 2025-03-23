import numpy as np
import pandas as pd
import pycaret
import transformers
from transformers import AutoModel, BertTokenizerFast
import matplotlib.pyplot as plt
#from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
# specify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Dataset
true_data = pd.read_csv('dataset/True.csv')
fake_data = pd.read_csv('dataset/Fake.csv')
# Generate labels True/Fake under new Target Column in 'true_data' and 'fake_data'
true_data['Target'] = ['True']*len(true_data)
fake_data['Target'] = ['Fake']*len(fake_data)
# Merge 'true_data' and 'fake_data', by random mixing into a single df called 'data'
data = pd.concat([true_data, fake_data], ignore_index=True).sample(frac=1).reset_index()
# Target column is made of string values True/Fake, let's change it to numbers 0/1 (Fake=1)
data['label'] = pd.get_dummies(data.Target)['Fake']
# Checking if our data is well balanced
label_size = [data['label'].sum(),len(data['label'])-data['label'].sum()]

# Train-Validation-Test set split into 70:15:15 ratio
# Train-Temp split
train_text, temp_text, train_labels, temp_labels = train_test_split(data['text'], data['label'],
                                                                    random_state=2018,
                                                                    test_size=0.3,
                                                                    stratify=data['Target'])
# Validation-Test split
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=2018,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

# Load BERT model and tokenizer via HuggingFace Transformers
bert = AutoModel.from_pretrained('bert')
tokenizer = BertTokenizerFast.from_pretrained('bert')

# Plot histogram of the number of words in train data 'title'
seq_len = [len(title.split()) for title in train_text]
pd.Series(seq_len).hist(bins = 40,color='firebrick')
plt.xlabel('Number of Words')
plt.ylabel('Number of texts')

# Majority of titles above have word length under 15. So, we set max title length as 32
MAX_LENGTH = 32
# Tokenize and encode sequences in the train set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = MAX_LENGTH,
    pad_to_max_length=True,
    truncation=True
)
# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = MAX_LENGTH,
    pad_to_max_length=True,
    truncation=True
)
# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = MAX_LENGTH,
    pad_to_max_length=True,
    truncation=True
)

# Convert lists to tensors and move to device
train_seq = torch.tensor(tokens_train['input_ids']).to(device)
train_mask = torch.tensor(tokens_train['attention_mask']).to(device)
train_y = torch.tensor(train_labels.tolist()).to(device)
val_seq = torch.tensor(tokens_val['input_ids']).to(device)
val_mask = torch.tensor(tokens_val['attention_mask']).to(device)
val_y = torch.tensor(val_labels.tolist()).to(device)
test_seq = torch.tensor(tokens_test['input_ids']).to(device)
test_mask = torch.tensor(tokens_test['attention_mask']).to(device)
test_y = torch.tensor(test_labels.tolist())

# Data Loader structure definition
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
batch_size = 32                                               #define a batch size
train_data = TensorDataset(train_seq, train_mask, train_y)    # wrap tensors
train_sampler = RandomSampler(train_data)                     # sampler for sampling the data during training
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
                                                              # dataLoader for train set
val_data = TensorDataset(val_seq, val_mask, val_y)            # wrap tensors
val_sampler = SequentialSampler(val_data)                     # sampler for sampling the data during training
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)
                                                              # dataLoader for validation set


# Unfreeze last 2 BERT layers instead of freezing all
for name, param in bert.named_parameters():
    if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

class BERT_Arch(nn.Module):
    def __init__(self, bert):
      super(BERT_Arch, self).__init__()
      self.bert = bert
      self.dropout = nn.Dropout(0.1)            # dropout layer
      self.relu =  nn.ReLU()                    # relu activation function
      self.fc1 = nn.Linear(768,512)             # dense layer 1
      self.fc2 = nn.Linear(512,2)               # dense layer 2 (Output layer)
      self.softmax = nn.LogSoftmax(dim=1)       # softmax activation function
    def forward(self, sent_id, mask):           # define the forward pass
      cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
                                                # pass the inputs to the model
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)                           # output layer
      x = self.softmax(x)                       # apply softmax activation
      return x
model = BERT_Arch(bert)
# Move model to GPU
model.to(device)
# Defining the hyperparameters (optimizer, weights of the classes and the epochs)
# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr = 1e-5)
# Define the loss function
cross_entropy  = nn.NLLLoss()
# Number of training epochs
epochs = 10


# Defining training and evaluation functions
def train():
    model.train()
    total_loss, total_accuracy = 0, 0

    for step, batch in enumerate(train_dataloader):  # iterate over batches
        if step % 50 == 0 and not step == 0:  # progress update after every 50 batches.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        # Move batch to device
        batch = [r.to(device) for r in batch]  # push the batch to gpu
        sent_id, mask, labels = batch
        model.zero_grad()  # clear previously calculated gradients
        preds = model(sent_id, mask)  # get model predictions for current batch
        loss = cross_entropy(preds, labels.long())  # compute loss between actual & predicted values
        total_loss = total_loss + loss.item()  # add on to the total loss
        loss.backward()  # backward pass to calculate the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       1.0)  # clip gradients to 1.0. It helps in preventing exploding gradient problem
        optimizer.step()  # update parameters
        preds = preds.detach().cpu().numpy()  # model predictions are stored on GPU. So, push it to CPU

    avg_loss = total_loss / len(train_dataloader)  # compute training loss of the epoch
    return avg_loss  # returns the loss and predictions


def evaluate():
    print("\nEvaluating...")
    model.eval()  # Deactivate dropout layers
    total_loss, total_accuracy = 0, 0
    for step, batch in enumerate(val_dataloader):  # Iterate over batches
        if step % 50 == 0 and not step == 0:  # Progress update every 50 batches.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
        # Move batch to device
        batch = [t.to(device) for t in batch]  # Push the batch to GPU
        sent_id, mask, labels = batch
        with torch.no_grad():  # Deactivate autograd
            preds = model(sent_id, mask)  # Model predictions
            loss = cross_entropy(preds, labels.long())  # Compute the validation loss between actual and predicted values
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
    avg_loss = total_loss / len(val_dataloader)  # compute the validation loss of the epoch
    return avg_loss


# Train and predict
best_valid_loss = float('inf')
train_losses = []  # empty lists to store training and validation loss of each epoch
valid_losses = []

for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    train_loss = train()  # train model
    valid_loss = evaluate()  # evaluate model
    if valid_loss < best_valid_loss:  # save the best model
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'c2_new_model_weights.pt')
    train_losses.append(train_loss)  # append training and validation loss
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

# load weights of best model
path = 'c2_new_model_weights.pt'
model.load_state_dict(torch.load(path))

with torch.no_grad():
  preds = model(test_seq, test_mask)
  preds = preds.detach().cpu().numpy()
  
preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds))