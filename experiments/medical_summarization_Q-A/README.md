# Medical Report Summarization

This project builds a fine-tuned T5 model to summarize medical reports, making them more accessible and understandable to patients and healthcare providers.

## Overview

Medical reports are often filled with technical language that can be difficult for patients to understand. This summarization model takes medical reports as input and generates concise, accessible summaries while preserving the key clinical information.

## Dataset

The project uses the Medical Meadow MediQA dataset, which contains:
- Medical reports as input text
- Corresponding simplified summaries as output text

The dataset is split into:
- Training set (80% of data)
- Validation set (10% of data)
- Test set (10% of data)

## Technical Implementation

### Base Model

- **Model**: Google's T5-small (text-to-text transformer)
- **Architecture**: Sequence-to-sequence model with encoder-decoder architecture
- **Optimization**: PEFT (Parameter-Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation)

### LoRA Configuration

The model uses parameter-efficient fine-tuning with these settings:
- `r=4`: Rank dimension for the low-rank decomposition
- `lora_alpha=1`: Scaling factor for weight matrices
- `target_modules=["q", "v"]`: Applying LoRA only to query and value matrices
- `lora_dropout=0.05`: Dropout rate to prevent overfitting
- `bias="lora_only"`: Training only the bias parameters related to LoRA
- `task_type="SEQ_2_SEQ_LM"`: Configured for sequence-to-sequence language modeling

### Training Approach

- **Input Formatting**: Prefixed with "summarize: " to direct the model
- **Max Input Length**: 1024 tokens
- **Max Output Length**: 128 tokens
- **Training Duration**: 3 epochs
- **Learning Rate**: 3e-2
- **Gradient Accumulation**: 4 steps
- **Mixed Precision**: FP16 for faster training
- **Evaluation Strategy**: Every 100 steps
- **Checkpoint Saving**: Every 200 steps

### Evaluation Metrics

The model performance is evaluated using:
- **ROUGE-1, ROUGE-2, ROUGE-L**: Measures overlap between generated and reference summaries
- **BLEU Score**: Measures precision of n-grams between generated and reference summaries

## Usage

### Requirements

Install the required packages:
```bash
pip install transformers datasets peft evaluate numpy python-dotenv
```

### Environment Setup

Create a `.env` file with your Hugging Face authentication token:
```
HF_TOKEN=your_huggingface_token_here
```

### Running the Code

1. Prepare the dataset in JSON format at `./src/data/medical_meadow_mediqa.json`
2. Run the script to train the model

### Inference Example

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the fine-tuned model
model = PeftModel.from_pretrained("path_to_saved_model")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

# Input text
medical_report = "Patient presents with..."

# Generate summary
input_text = "summarize: " + medical_report
inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
output = model.generate(**inputs, max_length=128)
summary = tokenizer.decode(output[0], skip_special_tokens=True)

print(summary)
```

## Project Structure

```
.
├── src/
│   ├── data/
│   │   └── medical_meadow_mediqa.json
├── results/            # Model checkpoints
├── logs/               # Training logs for TensorBoard
├── custom_metrics/     # Saved evaluation metrics
└── README.md
```

## Future Improvements

- Experiment with larger T5 variants for potentially better performance
- Implement domain-specific preprocessing for medical terminology
- Develop a user interface for easy access to the summarization tool
- Add more evaluation metrics specific to medical text quality
