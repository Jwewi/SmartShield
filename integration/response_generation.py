import warnings
import logging

# Suppress specific warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(filename='response_log.log', level=logging.INFO)

# Function to generate responses
def generate_response(threat_type):
    responses = {
        "phishing": "Implement email filtering and user training to prevent phishing attacks.",
        "malware": "Ensure your antivirus software is up to date and conduct regular scans.",
        "ransomware": "Regularly back up your data and avoid opening suspicious emails.",
        "DDoS": "Deploy DDoS protection solutions and monitor network traffic."
    }
    return responses.get(threat_type, "Monitor the network for any suspicious activities.")

# Example of using the function
detected_threats = ["phishing", "malware", "ransomware", "DDoS"]
for threat_type in detected_threats:
    response = generate_response(threat_type)
    logging.info(f"Detected Threat: {threat_type}, Recommended Response: {response}")
    print(f"Detected Threat: {threat_type}, Recommended Response: {response}")
