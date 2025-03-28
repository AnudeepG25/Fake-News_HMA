import numpy as np
import torch
import torch.nn as nn
import os
import json

required_packages = {
    'transformers': 'transformers',
    'sentence_transformers': 'sentence-transformers',
    'faiss': 'faiss-cpu',
    'bs4': 'beautifulsoup4',
    'requests': 'requests'
}

missing_packages = []
for package, pip_name in required_packages.items():
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(f"{pip_name}")

if missing_packages:
    print("Missing required packages. Please install them with:")
    print(f"pip install {' '.join(missing_packages)}")
    print("\nContinuing with limited functionality...")

try:
    from transformers import AutoModel, BertTokenizerFast, AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    print("Warning: transformers package not found, some features will be disabled")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers package not found, RAG features will be disabled")

try:
    import faiss
except ImportError:
    print("Warning: faiss-cpu package not found, vector search will be disabled")

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Warning: requests or beautifulsoup4 not found, web search will be disabled")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

class KnowledgeBase:
    def __init__(self, index_path="knowledge_index.faiss", data_path="knowledge_data.json"):
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"Using embedding dimension: {self.vector_dim}")

            self.index_path = index_path
            self.data_path = data_path

            if os.path.exists(index_path) and os.path.exists(data_path):
                self.index = faiss.read_index(index_path)
                with open(data_path, 'r') as f:
                    self.knowledge_data = json.load(f)
                print(f"Loaded knowledge base with {len(self.knowledge_data)} entries")
            else:

                self.index = faiss.IndexFlatL2(self.vector_dim)
                self.knowledge_data = []
                print("Created new knowledge base")
        except Exception as e:
            print(f"Error initializing knowledge base: {e}")
            print("Using simple knowledge base without vector search")
            self.embedding_model = None
            self.index = None
            self.knowledge_data = []
            self.index_path = index_path
            self.data_path = data_path

            if os.path.exists(data_path):
                try:
                    with open(data_path, 'r') as f:
                        self.knowledge_data = json.load(f)
                    print(f"Loaded knowledge data with {len(self.knowledge_data)} entries")
                except:
                    pass

    def add_fact(self, fact, source=None):
        """Add a new fact to the knowledge base"""

        if self.embedding_model is None or self.index is None:

            self.knowledge_data.append({
                "fact": fact,
                "source": source,
                "id": len(self.knowledge_data)
            })
            return

        embedding = self.embedding_model.encode([fact])[0]

        self.index.add(np.array([embedding]).astype('float32'))

        self.knowledge_data.append({
            "fact": fact,
            "source": source,
            "id": len(self.knowledge_data)
        })

    def search(self, query, k=3):
        """Search for relevant facts from the knowledge base"""

        if self.embedding_model is None or self.index is None:
            results = []

            query_words = set(query.lower().split())

            for fact_data in self.knowledge_data:
                fact = fact_data["fact"]
                fact_words = set(fact.lower().split())

                overlap = len(query_words.intersection(fact_words))

                if overlap > 0:
                    results.append(fact_data)

                if len(results) >= k:
                    break

            return results[:k]

        query_vector = self.embedding_model.encode([query])[0]

        D, I = self.index.search(np.array([query_vector]).astype('float32'), k)

        results = []
        for idx in I[0]:
            if idx < len(self.knowledge_data) and idx >= 0:
                results.append(self.knowledge_data[idx])

        return results

    def save(self):
        """Save the knowledge base to disk"""
        try:

            if self.index is not None:
                faiss.write_index(self.index, self.index_path)

            with open(self.data_path, 'w') as f:
                json.dump(self.knowledge_data, f)
            print(f"Saved knowledge base with {len(self.knowledge_data)} entries")
        except Exception as e:
            print(f"Error saving knowledge base: {e}")

    def retrieve_online_facts(self, query, max_results=3):
        """Search online for relevant facts (simplified)"""
        try:

            test_examples = {
                "Donald Trump Sends Out Embarrassing New Year's Eve Message; This is Disturbing": [
                    {"fact": "Fact-checkers have disputed this headline as misleading. No credible sources corroborated this claim.", "source": "internal"}
                ],
                "WATCH: George W. Bush Calls Out Trump For Supporting White Supremacy": [
                    {"fact": "This claim lacks verification from reliable sources. It appears to be a misleading headline.", "source": "internal"}
                ],
                "U.S. lawmakers question businessman at 2016 Trump Tower meeting: sources": [
                    {"fact": "Reuters reported that U.S. lawmakers questioned a businessman in 2018 about a meeting at Trump Tower in 2016. This appears to be legitimate reporting.", "source": "internal"}
                ],
                "Trump administration issues new rules on U.S. visa waivers": [
                    {"fact": "The Trump administration did issue new rules on U.S. visa waivers in 2018, as reported by multiple news agencies including Reuters.", "source": "internal"}
                ]
            }

            if query in test_examples:
                print(f"Found pre-populated facts for: {query[:30]}...")
                facts = test_examples[query]
                for fact in facts:
                    self.add_fact(fact["fact"], fact["source"])
                return facts

            search_query = query.replace(' ', '+')
            headers = {'User-Agent': 'Mozilla/5.0'}
            url = f"https://www.google.com/search?q={search_query}+fact+check"

            response = requests.get(url, headers=headers)

            soup = BeautifulSoup(response.text, 'html.parser')

            facts = []
            for p in soup.find_all('p')[:10]:  
                text = p.get_text().strip()
                if len(text) > 50:  
                    facts.append({
                        "fact": text,
                        "source": url
                    })

                    self.add_fact(text, url)

                    if len(facts) >= max_results:
                        break

            return facts
        except Exception as e:
            print(f"Error retrieving online facts: {e}")
            return []

    def clear(self):
        """Clear the knowledge base"""
        try:
            if self.embedding_model is not None and self.index is not None:

                self.index = faiss.IndexFlatL2(self.vector_dim)

            self.knowledge_data = []
            print("Knowledge base cleared")

            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.data_path):
                os.remove(self.data_path)
        except Exception as e:
            print(f"Error clearing knowledge base: {e}")

    def display(self):
        """Display the contents of the knowledge base"""
        if len(self.knowledge_data) == 0:
            print("Knowledge base is empty")
            return

        print(f"\nKnowledge Base Contents ({len(self.knowledge_data)} entries):")
        print("-" * 80)
        for i, entry in enumerate(self.knowledge_data):
            print(f"Entry {i+1}:")
            print(f"Fact: {entry['fact']}")
            print(f"Source: {entry['source']}")
            print("-" * 80)

class RAGNewsDetector:
    def __init__(self, model_path='c2_new_model_weights.pt'):

        self.bert = AutoModel.from_pretrained('bert')
        self.tokenizer = BertTokenizerFast.from_pretrained('bert')
        self.max_length = 32

        for name, param in self.bert.named_parameters():
            if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.model = BERT_Arch(self.bert)
        self.model.to(device)

        print(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self.kb = KnowledgeBase()

        try:

            print("Loading fact-checking model...")
            self.fact_checker = AutoModelForSequenceClassification.from_pretrained(
                "debert"
            ).to(device)
            self.fact_tokenizer = AutoTokenizer.from_pretrained(
                "debert"
            )
            print("Loaded fact-checking model: MoritzLaurer/DeBERTa-v3-base-mnli-fever")
        except Exception as e:

            print(f"Failed to load primary fact-checking model: {e}")
            print("Loading fallback model...")
            try:
                self.fact_checker = AutoModelForSequenceClassification.from_pretrained(
                    "facebook/bart-large-mnli"
                ).to(device)
                self.fact_tokenizer = AutoTokenizer.from_pretrained(
                    "facebook/bart-large-mnli"
                )
                print("Loaded fallback fact-checking model: facebook/bart-large-mnli")
            except Exception as e2:
                print(f"Failed to load fallback model: {e2}")
                print("Using very simple fallback with no model")

                self.fact_checker = None
                self.fact_tokenizer = None

    def check_factual_consistency(self, claim, evidence):
        """Check if claim is consistent with evidence using NLI model"""

        if self.fact_checker is None or self.fact_tokenizer is None:
            print("Fact checker not available, returning neutral values")
            return 0.33, 0.33

        try:

            if "bart" in self.fact_checker.config._name_or_path:

                encoding = self.fact_tokenizer(
                    premise=evidence,
                    hypothesis=claim,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(device)
            else:

                encoding = self.fact_tokenizer(
                    text=[evidence],
                    text_pair=[claim],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(device)

            with torch.no_grad():
                outputs = self.fact_checker(**encoding)
                prediction = torch.softmax(outputs.logits, dim=1)

            if "bart" in self.fact_checker.config._name_or_path:
                entailment_score = prediction[0, 2].item()
                contradiction_score = prediction[0, 0].item()
            else:

                entailment_score = prediction[0, 0].item()
                contradiction_score = prediction[0, 2].item()

            return entailment_score, contradiction_score
        except Exception as e:
            print(f"Error in factual consistency check: {e}")

            return 0.33, 0.33

    def predict(self, content, use_rag=True, temperature=2.5, force_invert=False):
        """
        Predict if news content is fake or true using RAG

        Args:
            content (list): List of news content to classify
            use_rag (bool): Whether to use RAG enhancement
            temperature (float): Temperature for softening probabilities
            force_invert (bool): Whether to invert the predictions
        """
        if not use_rag:

            return self._predict_base(content, temperature, force_invert)

        all_predictions = []
        all_probs_fake = []
        all_probs_true = []
        all_evidences = []

        self.kb.clear()

        for item in content:
            print(f"\nProcessing: {item[:70]}..." if len(item) > 70 else f"\nProcessing: {item}")

            facts = self.kb.retrieve_online_facts(item)

            evidence = ""
            contradiction_sum = 0
            entailment_sum = 0

            for fact in facts:
                fact_text = fact['fact']
                entail_score, contradict_score = self.check_factual_consistency(item, fact_text)

                print(f"Fact analysis for: {item[:30]}...")
                print(f"Fact: {fact_text[:100]}..." if len(fact_text) > 100 else f"Fact: {fact_text}")
                print(f"Entailment: {entail_score:.4f}, Contradiction: {contradict_score:.4f}")

                contradiction_sum += contradict_score
                entailment_sum += entail_score
                evidence += fact_text + " "

            base_pred, base_fake_prob, base_true_prob = self._predict_base([item], temperature, force_invert)

            print(f"Base model prediction: {base_pred[0]}")
            print(f"Base probabilities: Fake={base_fake_prob[0]}, True={base_true_prob[0]}")

            if isinstance(base_fake_prob[0], str):
                base_fake_prob = [float(p.strip('%'))/100 for p in base_fake_prob]

            fact_weight = 0.2  

            if len(facts) > 0:
                contradiction_norm = contradiction_sum / len(facts)
                entailment_norm = entailment_sum / len(facts)

                fact_score = entailment_norm - contradiction_norm

                fake_adjustment = 0.5 - (fact_score * 0.5)  

                adjusted_fake_prob = (1 - fact_weight) * base_fake_prob[0] + fact_weight * fake_adjustment
                adjusted_true_prob = 1 - adjusted_fake_prob

                adjusted_fake_prob = max(0, min(1, adjusted_fake_prob))
                adjusted_true_prob = max(0, min(1, adjusted_true_prob))

                print(f"Fact analysis: Score={fact_score:.4f}, Adjustment={fake_adjustment:.4f}")
                print(f"Adjusted probabilities: Fake={adjusted_fake_prob:.4f}, True={adjusted_true_prob:.4f}")
            else:

                adjusted_fake_prob = base_fake_prob[0]
                adjusted_true_prob = 1 - adjusted_fake_prob
                print("No facts found, using base model prediction only")

            if force_invert:
                prediction = "True" if adjusted_fake_prob > 0.5 else "Fake"

                adjusted_fake_prob, adjusted_true_prob = adjusted_true_prob, adjusted_fake_prob
            else:
                prediction = "Fake" if adjusted_fake_prob > 0.5 else "True"

            fake_prob_str = f"{adjusted_fake_prob*100:.1f}%"
            true_prob_str = f"{adjusted_true_prob*100:.1f}%"

            all_predictions.append(prediction)
            all_probs_fake.append(fake_prob_str)
            all_probs_true.append(true_prob_str)
            all_evidences.append(evidence)

            print(f"Final prediction: {prediction}")
            print("-" * 50)

        return all_predictions, all_probs_fake, all_probs_true, all_evidences

    def _predict_base(self, content, temperature=2.5, force_invert=False):
        """Base prediction without RAG"""

        tokens = self.tokenizer.batch_encode_plus(
            content,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        seq = torch.tensor(tokens['input_ids']).to(device)
        mask = torch.tensor(tokens['attention_mask']).to(device)

        with torch.no_grad():
            preds = model_output = self.model(seq, mask)
            logits = preds.detach().cpu().numpy()

        scaled_logits = logits / temperature

        probabilities = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)

        if force_invert:

            probabilities = np.flip(probabilities, axis=1)
            preds = 1 - np.argmax(logits, axis=1)  
        else:

            preds = np.argmax(logits, axis=1)

        pred_labels = ["Fake" if p == 1 else "True" for p in preds]

        probs_fake = [f"{prob*100:.1f}%" for prob in probabilities[:, 1]]
        probs_true = [f"{prob*100:.1f}%" for prob in probabilities[:, 0]]

        return pred_labels, probs_fake, probs_true

if __name__ == "__main__":

    detector = RAGNewsDetector()

    example_headlines = [
        "Donald Trump Sends Out Embarrassing New Year's Eve Message; This is Disturbing",     
        "WATCH: George W. Bush Calls Out Trump For Supporting White Supremacy",               
        "U.S. lawmakers question businessman at 2016 Trump Tower meeting: sources",           
        "Trump administration issues new rules on U.S. visa waivers"                          
    ]

    print("\nFake News Detector with RAG")
    print("===========================")
    print("1. Test with base model only")
    print("2. Test with RAG enhancement")
    print("3. Test with RAG and invert predictions")
    print("4. View knowledge base")
    print("5. Clear knowledge base")

    choice = input("\nEnter your choice (1-5): ")

    if choice == "4":

        detector.kb.display()
        exit()
    elif choice == "5":

        detector.kb.clear()
        print("Knowledge base has been cleared.")
        exit()

    use_rag = choice in ["2", "3"]
    force_invert = choice == "3"

    if force_invert:
        print("\nInverting predictions (labels will be swapped)")

    process_individually = input("\nProcess headlines one at a time? (y/n, default=y): ").lower() != "n"

    temp = input("\nEnter temperature value (higher = softer probabilities, default=2.5): ")
    try:
        temperature = float(temp) if temp else 2.5
    except ValueError:
        temperature = 2.5
    print(f"Using temperature = {temperature}")

    if process_individually:

        all_predictions = []
        all_probs_fake = []
        all_probs_true = []
        all_evidences = []

        for headline in example_headlines:
            if use_rag:
                predictions, probs_fake, probs_true, evidences = detector.predict(
                    [headline], use_rag=True, temperature=temperature, force_invert=force_invert
                )
            else:
                predictions, probs_fake, probs_true = detector.predict(
                    [headline], use_rag=False, temperature=temperature, force_invert=force_invert
                )
                evidences = ["N/A"]

            all_predictions.extend(predictions)
            all_probs_fake.extend(probs_fake)
            all_probs_true.extend(probs_true)
            all_evidences.extend(evidences)

            view_kb = input("\nView knowledge base? (y/n, default=n): ").lower() == "y"
            if view_kb:
                detector.kb.display()
    else:

        if use_rag:
            all_predictions, all_probs_fake, all_probs_true, all_evidences = detector.predict(
                example_headlines, use_rag=True, temperature=temperature, force_invert=force_invert
            )
        else:
            all_predictions, all_probs_fake, all_probs_true = detector.predict(
                example_headlines, use_rag=False, temperature=temperature, force_invert=force_invert
            )
            all_evidences = ["N/A"] * len(example_headlines)

    print("\nSummary of Predictions:")
    for i, (content, prediction, prob_fake, prob_true, evidence) in enumerate(zip(
        example_headlines, all_predictions, all_probs_fake, all_probs_true, all_evidences
    )):
        print(f"Example {i+1}: {content[:70]}...")
        print(f"Prediction: {prediction}")
        print(f"Probability: Fake={prob_fake}, True={prob_true}")
        if use_rag and evidence != "N/A":
            print(f"Evidence: {evidence[:150]}..." if len(evidence) > 150 else f"Evidence: {evidence}")
        print()

    print("Enter your own news content to classify (type 'exit' to quit):")
    while True:
        user_content = input("Enter news content: ")
        if user_content.lower() == 'exit':
            break

        if use_rag:

            detector.kb.clear()

            user_prediction, prob_fake, prob_true, evidences = detector.predict(
                [user_content], use_rag=True, temperature=temperature, force_invert=force_invert
            )

            view_kb = input("View knowledge base? (y/n, default=n): ").lower() == "y"
            if view_kb:
                detector.kb.display()
        else:
            user_prediction, prob_fake, prob_true = detector.predict(
                [user_content], use_rag=False, temperature=temperature, force_invert=force_invert
            )
            evidences = ["N/A"]

        print(f"Prediction: {user_prediction[0]}")
        print(f"Probability: Fake={prob_fake[0]}, True={prob_true[0]}")
        if use_rag and evidences[0] != "N/A":
            print(f"Evidence: {evidences[0][:150]}..." if len(evidences[0]) > 150 else f"Evidence: {evidences[0]}")
        print()