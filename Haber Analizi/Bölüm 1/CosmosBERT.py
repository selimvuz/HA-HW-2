import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Model ve Tokenizer yükle
print("Model yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(
    "ytu-ce-cosmos/turkish-base-bert-uncased")
model = AutoModel.from_pretrained("ytu-ce-cosmos/turkish-base-bert-uncased")

# GPU kullanılabilirse modeli aktar
# Bu kısmı yoruma alıyorum, gerektiğinde aktive edebilirsiniz
# if torch.cuda.is_available():
#     model = model.cuda()
#     print("Model GPU'ya aktarıldı.")

def get_embeddings(text, index, total):
    print(f"İşlem: {index+1}/{total}...")
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach()
    return embeddings.numpy()

def load_data(directory):
    texts, labels = [], []
    categories = ["ekonomi", "magazin", "siyaset", "saglik", "spor"]
    label_dict = {category: i for i, category in enumerate(categories)}
    for category in categories:
        dir_path = os.path.join(directory, category)
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r', encoding='ISO-8859-1') as file:
                texts.append(file.read())
                labels.append(label_dict[category])
    return texts, labels

# Veri kümesini yükleyin ve etiketlerini ayarlayın
print("Veri kümesi yükleniyor...")
texts, labels = load_data('../../datasets/news')

# Toplam metin sayısını hesapla
total_texts = len(texts)

# Temsilleri elde et
print("Vektör temsilleri elde ediliyor...")
embeddings = np.vstack([get_embeddings(text, idx, total_texts)
                       for idx, text in enumerate(texts)])

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42)

# Her bir sınıflandırıcı için eğit ve değerlendir
classifiers = [
    ("Random Forest", RandomForestClassifier()),
    ("SVM", SVC()),
    ("Logistic Regression", LogisticRegression(max_iter=1000))
]

for name, clf in classifiers:
    scores = cross_val_score(clf, embeddings, labels, cv=5)
    print(f"{name} 5-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)" %
          (scores.mean(), scores.std() * 2))
