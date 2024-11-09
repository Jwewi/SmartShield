import warnings
import logging
import pandas as pd
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

# Load the dataset
dataset_path = '../dataset/cleaned_dataset.xls'  # Update with the correct relative path to your CSV file
df = pd.read_csv(dataset_path)
print(df.head())  # Display the first few rows of the dataset

# Create a new DataFrame to store predictions and responses
predictions_df = pd.DataFrame(columns=['Context', 'Prediction', 'Response'])

# Iterate through the dataset and make predictions
for index, row in df.iterrows():
    sample_text = f"{row['context']} [MASK]."  # Adjust this line to match your dataset structure
    results = pipe(sample_text)
    if results:
        top_result = results[0]
        threat_type = top_result['token_str']
        response = generate_response(threat_type)
        predictions_df = predictions_df.append({
            'Context': row['context'],
            'Prediction': threat_type,
            'Response': response
        }, ignore_index=True)

# Display the new DataFrame with predictions and responses
print(predictions_df.head())

# Save the predictions and responses to a separate CSV file
predictions_df.to_csv('../dataset/predictions_responses.csv', index=False)
