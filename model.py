# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("fill-mask", model="jackaduma/SecBERT")
print(pipe("Cybersecurity threats are [MASK]."))

# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("jackaduma/SecBERT")
model = AutoModelForMaskedLM.from_pretrained("jackaduma/SecBERT")

# Example input
inputs = tokenizer("Cybersecurity threats are [MASK].", return_tensors="pt")
outputs = model(**inputs)
print(outputs)

