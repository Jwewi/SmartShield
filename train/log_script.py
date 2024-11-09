import warnings
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM, logging as transformers_logging

# Suppress specific warnings
warnings.filterwarnings("ignore", message="BertForMaskedLM has generative capabilities")
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")

# Configure logging
logging.basicConfig(filename='model_output.log', level=logging.INFO)

# Set transformers logging to ERROR to suppress detailed warnings
transformers_logging.set_verbosity_error()

# Load tokenizer and model using auto classes
tokenizer = AutoTokenizer.from_pretrained("jackaduma/SecBERT", trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained("jackaduma/SecBERT", trust_remote_code=True)

# Use the pipeline with SecBERT model
pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Sample sentence with a [MASK] token
sample_text = "The network detected a [MASK] attack."

# Get predictions
results = pipe(sample_text)

# Log the results
logging.info(results)

# Display results
for result in results:
    print(f"Token: {result['token_str']}, Score: {result['score']}, Sequence: {result['sequence']}")
