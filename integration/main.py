import warnings
import logging
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM, logging as transformers_logging
from response_generation import generate_response
from azure.loganalytics import LogAnalyticsDataClient
from azure.loganalytics.models import QueryBody
from azure.identity import DefaultAzureCredential
import json

# Suppress specific warnings
warnings.filterwarnings("ignore", message="BertForMaskedLM has generative capabilities")
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")

# Configure logging
logging.basicConfig(filename='model_output.log', level=logging.INFO)

# Set transformers logging to ERROR to suppress detailed warnings
transformers_logging.set_verbosity_error()

# Load the model and tokenizer
model = AutoModelForMaskedLM.from_pretrained("jackaduma/SecBERT", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("jackaduma/SecBERT", trust_remote_code=True)

# Use the pipeline with SecBERT model
pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Load the dataset
dataset_path = '../dataset/cleaned_dataset.csv'  # Update with the correct relative path to your CSV file
df = pd.read_csv(dataset_path)
print(df.head())  # Display the first few rows of the dataset

# Check the column names
print("Columns in the dataset:", df.columns)

# Create a new DataFrame to store predictions and responses
predictions_df = pd.DataFrame(columns=['Context', 'Prediction', 'Response'])

# Ensure 'context' column exists in the DataFrame
if 'context' not in df.columns:
    df['context'] = ''  # Initialize the 'context' column with empty strings or appropriate default values

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
predictions_df.to_csv('../dataset/cleaned_datasetresponses.csv', index=False)

# Send data to Microsoft Sentinel
workspace_id = '8907f303-1f30-422f-8f86-6634cb28bcc6'  # Replace with your Log Analytics workspace ID
primary_key = 'C6MN0Q8X3FPLniqCQ13pfKJLSSfP8MRY90d0VTenECxoOllvbvZhoiPZflxOBFToGy4xoHt6f5bA2d6rz29akA=='  # Replace with your Log Analytics primary key

# Create a client
client = LogAnalyticsDataClient(credential=DefaultAzureCredential())

# Prepare the data to send
data = predictions_df.to_dict(orient='records')
body = json.dumps(data)

# Send the data
response = client.query(workspace_id, QueryBody(query=body))
print(response)
