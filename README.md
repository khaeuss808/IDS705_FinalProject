# IDS705 Final Project: Text Classification for Medical Specialty

## 🔍 Project Overview
Efficient patient triage is a longstanding challenge in healthcare. Our project develops a transformer-based NLP pipeline that classifies patient-reported symptoms into medical specialties to streamline clinical intake, reduce misdiagnoses, and improve referral accuracy.

We fine-tuned and evaluated various BERT-family models on a unified dataset of symptom descriptions. Our system also explores robustness to noise, cross-lingual generalization, embedding interpretability, and integration into an end-to-end clinical workflow.

## 🎯 Objectives
- Classify patient symptom text into 8 medical specialties using transformer models.
- Evaluate robustness under adversarial noise (insertions and spelling errors).
- Test cross-lingual generalization (English → Chinese).
- Visualize model embeddings for interpretability.
- Deploy the model in a simplified medical intake framework with summarization and Q&A support.

## 🗃️ Dataset
We curated and consolidated data from the following sources:
- [Medical Speech Transcription and Intent (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent)
- [Symptom2Disease (Kaggle)](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)
- [Diseases_Symptoms (HuggingFace)](https://huggingface.co/datasets/QuyenAnhDE/Diseases_Symptoms)

Final dataset: ~2,300 rows of patient symptom text mapped into 8 unified condition categories.

Additional datasets were used for:
- Robustness testing (synthetic noise generation)
- Chinese translation (via `deep_translator`)
- Summarization and Q&A (e.g., MedAlpaca, MedSum)

## 🧪 Experiments
| No. | Name | Description |
|----|------|-------------|
| 1 | Pre-trained Model Comparison | Compared BERT-base, RoBERTa-base, and DistilBERT for multi-class symptom classification |
| 2 | Robustness Testing | Introduced spelling/insertion noise to test model stability |
| 3 | Cross-Lingual Evaluation | Tested English-trained models on Chinese inputs using mBERT and Chinese-BERT |
| 4 | Embedding Analysis | t-SNE, SHAP, and attention visualization to interpret model decisions |
| 5 | Medical Framework Integration | Built an end-to-end pipeline with summarization (T5) and Q&A (GPT2) modules |

## 📁 Repository Structure
| File                                     | Description                                                      |
|------------------------------------------|------------------------------------------------------------------|
| `data_manipulation.ipynb`                | Consolidation of three datasets for use in all experiments       |
| `experiment1.ipynb`                      | Model comparison (BERT, RoBERTa, DistilBERT)                     |
| `adversarial_testing_zihan.ipynb`        | Robustness evaluation with noisy inputs                          |
| `experiment3.ipynb`                      | Zero-shot and fine-tuned cross-lingual evaluation                |
| `updated_experiment_4_embedding.ipynb`   | Embedding analysis and interpretability                          |
| `final_data.csv`                         | Final dataset                                                    |
| `adversarial_results.csv`                | Results from adversarial tests                                   |
| `original_datasets/`                     | Raw symptom dataset prior to consolidation                       |
| `medical_summarization_Q-A/`             | T5 model for summarizing medical reports with LoRA fine-tuning   |
| `README.md`                              | Project documentation (you're reading it!)                       |


## 🧠 Models Used
- Classification: `bert-base-uncased`, `roberta-base`, `distilbert-base-uncased`
- Multilingual: `bert-base-multilingual-cased`, `bert-base-chinese`
- Summarization: `T5-small`
- Q&A: `GPT-2` (fine-tuned)

## 🛠️ Environment Setup & Usage Instructions

## 🔧 Usage Guide for experiment3

### Requirements

Install the necessary dependencies:

```bash
pip install transformers datasets scikit-learn pandas matplotlib deep-translator
```
### Environment Setup

Make sure you have access to a Hugging Face token if required:

```bash
export HF_TOKEN=your_huggingface_token_here
```
## 📓 Running the Notebook

### Data Preparation

Ensure you have a CSV file with the following columns:

- `phrase`: English input text  
- `translated_phrase`: Machine-translated Chinese version  
- `label`: Corresponding numeric label or class name  

---

### Notebook Execution

Open and run all cells in `experiment3.ipynb`.

This notebook performs the following:

- Fine-tuning and zero-shot evaluation on BERT-base, Multilingual BERT, and Chinese-BERT  
- Translation preprocessing (if needed)  
- Accuracy, F1 score, and confusion matrix reporting  

---
### 🔍 Sample Code Snippet for Evaluation

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer from the saved output path
model_path = "./bert_output"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Input: Chinese symptom description
text = "我咳嗽发烧"

# Preprocess and predict
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()

print("Predicted class ID:", predicted_class)
```

## 📊 Key Results
- **RoBERTa-base** achieved the highest F1 score: **0.918**
- **Multilingual-BERT** (zero-shot) performed well in Chinese: **92.9% accuracy**
- **Chinese-BERT** (fine-tuned) outperformed others: **94.0% accuracy**
- Robustness under noise: **81.3%** (insert) and **95.8%** (spelling)
- Summarization ROUGE-L: **0.768**, Q&A accuracy: **25%**

## ⚖️ Ethical Considerations
The system is intended **only as a pre-screening tool**. It is not a substitute for medical diagnosis. Limitations in data quality, potential bias, and lack of explainability are acknowledged. Privacy concerns are mitigated through anonymized data, and SHAP/attention mechanisms support interpretability.

## 👥 Team & Contributions
- **Alejandro Paredes La Torre**: Medical summarization & Q&A (Exp 5)
- **Kayla Haeussler**: Data consolidation, Pre-trained Model Comparison(Exp 1)
- **Ramil Mammadov**: Embedding analysis and interpretability (Exp 4)
- **Sizhe Chen**: Cross-lingual evaluation (Exp 3), conclusion synthesis
- **Zihan Xiao**: Robustness testing (Exp 2), visualizations, section writing



