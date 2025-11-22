# 🧾 Transaction Categorization System
A hybrid ML + taxonomy-based system for categorizing raw noisy UPI/bank transaction texts into spending categories such as **shopping, utilities, travel, food**, etc. Supports **admin-managed taxonomy**, **user feedback loop**, and **auto retraining** to continuously improve accuracy.

---

## ✨ Features
- Fuzzy merchant matching with multi-token scoring
- ML fallback classifier (when merchant lookup confidence is low)
- Admin UI for managing taxonomy & synonyms
- User feedback UI for correcting predictions
- Automated retraining based on feedback or taxonomy updates
- Merchant suggestions ranking from user feedback

---

## 📂 Project Structure

```
Hackathon/
│
├── config/
│   ├── taxonomy.yaml
│   └── backups/
│
├── data/
│   ├── predict_transaction.py
│   ├── retrain_with_feedback.py
│   ├── user_feedback.csv
│   ├── merchant_suggestions.csv
│   └── taxonomy_synonym_seeds.csv
│
├── models/
│   ├── baseline_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── tools/
│   ├── merchant_lookup.py
│   ├── taxonomy.py
│   ├── taxonomy_admin.py
│   └── user_feedback_ui.py
│
├── logs/
│   └── retrain_stdout.log
│
├── requirements.txt
└── README.md
```

---

## 🛠 Installation

```bash
pip install -r requirements.txt
```

---

## ▶ Run Prediction

```bash
python -m data.predict_transaction
```

Example:
```
Processing transaction: "UPI to nykaa@oksbi"
Merchant lookup → shopping score=0.97 span=nykaa
Using merchant-based classification
```

---

## 🙋 User Feedback UI

```bash
streamlit run tools/user_feedback_ui.py
```

### User Flow
1. Paste raw transaction lines
2. Click **Predict & Review**
3. Edit incorrect category
4. Enable **Accept**
5. Submit → saves to:
```
data/user_feedback.csv
data/merchant_suggestions.csv
```
Retraining triggers automatically.

---

## 🧠 Admin Taxonomy UI

```bash
streamlit run tools/taxonomy_admin.py
```

Admin Can:
| Action | Result |
|--------|--------|
Add category | Adds new spending category
Add merchant | Adds aliases & improves recognition
Upload taxonomy | Replace YAML and backup older version
Approve suggestions | Merchant suggestions → taxonomy
Retrain system | Automated

Files updated:
```
config/taxonomy.yaml
data/taxonomy_synonym_seeds.csv
logs/retrain_stdout.log
```

---

## 🔁 Learning Loop

### Prediction flow:
```
Text → MerchantLookup → high score? → Category
                         |
                         └ ML fallback → Category
```

### User improvement loop:
```
Correct categories → feedback.csv → train → improved model
```

### Admin improvement loop:
```
Suggestions.csv → approve → taxonomy.yaml → train
```

---

## 🧪 Example Test Input
```
Payment upi to amzn@upi
Electricity bill bescom
Online order mynro
upi transfer to randomNameXYZ
```

Expected:
- amzn → shopping
- bescom → utilities
- mynro → shopping (fuzzy)
- random → fallback model low confidence → review required

---

## 🏁 Future Upgrade Options
- Sentence-transformer embeddings for merchant detection
- Confidence curve & visualization dashboards
- Auto-rule approval when multiple users agree

---

## 📜 License
Open source prototype for academic/hackathon use.

