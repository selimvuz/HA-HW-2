import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Model ve Tokenizer yükle
tokenizer = AutoTokenizer.from_pretrained("Trendyol/Trendyol-LLM-7b-chat-v0.1")
model = AutoModel.from_pretrained("Trendyol/Trendyol-LLM-7b-chat-v0.1")

# GPU kullanılabilirse modeli aktar
if torch.cuda.is_available():
    model = model.cuda()

def get_embeddings(text, index, total):
    print(f"İşlem: {index+1}/{total}...")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach()
    if torch.cuda.is_available():
        embeddings = embeddings.cpu()  # GPU'dan CPU'ya taşı
    return embeddings.numpy()

def load_data(directory):
    texts, labels = [], []
    label_dict = {"pozitif": 0, "negatif": 1, "tarafsiz": 2}
    for label in label_dict:
        dir_path = os.path.join(directory, label)
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
                labels.append(label_dict[label])
    return texts, labels

# Veri kümesini yükleyin ve etiketlerini ayarlayın
texts, labels = load_data('datasets/sentiment')

# Toplam soru sayısını hesapla
total_texts = len(texts)

# 'text' için embeddingler elde et
embeddings = np.vstack([get_embeddings(text, idx, total_texts) for idx, text in enumerate(texts)])

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Sınıflandırıcıları tanımla
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression()
}

# Her bir sınıflandırıcı için eğit ve değerlendir
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{name} Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))
