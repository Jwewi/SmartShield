import warnings
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM, logging as transformers_logging
from response_generation import generate_response

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
for result in results:
    threat_type = result['token_str']
    response = generate_response(threat_type)
    logging.info(f"Detected Threat: {threat_type}, Recommended Response: {response}")
    print(f"Detected Threat: {threat_type}, Recommended Response: {response}")

# Load the dataset df = pd.read_csv('path_to_your_dataset.csv') # Update with the correct path to your CSV file print(df.head()) # Display the first few rows of the dataset