import numpy as np
import torch
from transformers import AutoModel, BertTokenizerFast
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BERT_Arch(nn.Module):
    def _init_(self, bert):
        super(BERT_Arch, self)._init_()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

for name, param in bert.named_parameters():
    if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model = BERT_Arch(bert)
model.to(device)

model_path = 'c2_new_model_weights.pt'
print(f"Loading model from {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  

MAX_LENGTH = 32

def predict_fake_news(content, is_full_text=False, force_invert=False):
    """
    Predict if news content is fake or true

    Args:
        content (list): List of news content to classify
        is_full_text (bool): Whether the content is full text
        force_invert (bool): Whether to force invert the predictions

    Returns:
        tuple: Predictions and confidence information
    """

    tokens = tokenizer.batch_encode_plus(
        content,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True
    )

    seq = torch.tensor(tokens['input_ids']).to(device)
    mask = torch.tensor(tokens['attention_mask']).to(device)

    with torch.no_grad():
        preds = model(seq, mask)
        logits = preds.detach().cpu().numpy()

    temperature = 2.5
    scaled_logits = logits / temperature

    probabilities = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)

    if force_invert:

        preds = np.argmax(1 - probabilities, axis=1)
    else:
        preds = np.argmax(probabilities, axis=1)

    pred_labels = ["Fake" if p == 1 else "True" for p in preds]

    probs_fake = [prob[1] * 100 for prob in probabilities]
    probs_true = [f"{prob[0] * 100:.1f}%" for prob in probabilities]

    return pred_labels, probs_fake, probs_true

if _name_ == "_main_":

    example_headlines = [
        "Donald Trump Sends Out Embarrassing New Year's Eve Message; This is Disturbing",  
        "WATCH: George W. Bush Calls Out Trump For Supporting White Supremacy",  
        "U.S. lawmakers question businessman at 2016 Trump Tower meeting: sources",  
        "Trump administration issues new rules on U.S. visa waivers"  
    ]

    print("\nHow would you like to test the model?")
    print("1. Test with headlines")
    print("2. Test with full text articles")
    print("3. Invert predictions (if model predicts opposite labels)")

    choice = input("\nEnter your choice (1/2/3): ")
    force_invert = False
    is_full_text = False

    if choice == "2":
        is_full_text = True
        print("\nPlease enter full text of news articles when prompted.")
    elif choice == "3":
        force_invert = True
        print("\nInverting predictions (True will be shown as Fake and vice versa).")

    predictions, probs_fake, probs_true = predict_fake_news(
        example_headlines, is_full_text=is_full_text, force_invert=force_invert
    )

    print("\nPredictions on example news content:")
    for i, (content, prediction, prob_fake, prob_true) in enumerate(zip(
            example_headlines, predictions, probs_fake, probs_true
    )):
        print(f"Example {i + 1}: {content[:70]}...")
        print(f"Prediction: {prediction}")
        print(f"Probability: Fake={prob_fake:.1f}%, True={prob_true}")
        print()

    print("Enter your own news content to classify (type 'exit' to quit):")
    while True:
        user_content = input("Enter news content: ")
        if user_content.lower() == 'exit':
            break

        user_prediction, prob_fake, prob_true = predict_fake_news(
            [user_content], is_full_text=is_full_text, force_invert=force_invert
        )
        print(f"Prediction: {user_prediction[0]}")
        print(f"Probability: Fake={prob_fake[0]:.1f}%, True={prob_true[0]}")
        print()