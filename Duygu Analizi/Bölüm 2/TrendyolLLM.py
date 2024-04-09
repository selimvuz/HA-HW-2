import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score

# Model ve Tokenizer yükle
print("Model yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained("Trendyol/Trendyol-LLM-7b-chat-v0.1")
model = AutoModel.from_pretrained("Trendyol/Trendyol-LLM-7b-chat-v0.1")

# GPU kullanılabilirse modeli aktar
"""
if torch.cuda.is_available():
    model = model.cuda()
    print("Model GPU'ya aktarıldı.")
"""


def get_embeddings(text, index, total):
    print(f"İşlem: {index+1}/{total}...")
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True, max_length=512)
    # if torch.cuda.is_available():
    #     inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach()
    # if torch.cuda.is_available():
    #     embeddings = embeddings.cpu()  # GPU'dan CPU'ya taşı
    return embeddings.numpy()


def load_data(directory):
    texts, labels = [], []
    label_dict = {"pozitif": 0, "negatif": 1, "tarafsiz": 2}
    for label in label_dict:
        dir_path = os.path.join(directory, label)
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r', encoding='ISO-8859-1') as file:
                texts.append(file.read())
                labels.append(label_dict[label])
    return texts, labels


# Veri kümesini yükleyin ve etiketlerini ayarlayın
print("Veri kümesi yükleniyor...")
texts, labels = load_data('../../datasets/sentiment')

# Toplam soru sayısını hesapla
total_texts = len(texts)

# Temsilleri elde et
print("Vektör temsilleri elde ediliyor...")
embeddings = np.vstack([get_embeddings(text, idx, total_texts)
                       for idx, text in enumerate(texts)])

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42)

# Belirli cümleler için temsilcileri (embeddings) al
reference_texts = [
    "Bu metin olumlu ifadeler içermektedir.",
    "Bu metin olumsuz ifadeler içermektedir.",
    "Bu metin tarafsız ifadeler içermektedir."
]
reference_embeddings = np.vstack([get_embeddings(text, idx, len(reference_texts))
                                  for idx, text in enumerate(reference_texts)])


def predict_with_cosine_similarity(embeddings, reference_embeddings):
    predictions = []
    for embedding in embeddings:
        # Her bir referans temsili ile kosinüs benzerliğini hesapla
        cosine_similarities = [
            1 - cosine(embedding, ref_emb) for ref_emb in reference_embeddings]
        # En yüksek benzerliğe sahip sınıfın indeksini bul
        predicted_class = np.argmax(cosine_similarities)
        predictions.append(predicted_class)
    return predictions


# Tahminleri yap
predictions = predict_with_cosine_similarity(X_test, reference_embeddings)

# Performansı değerlendir
print("Doğruluk Skoru:", accuracy_score(y_test, predictions))
