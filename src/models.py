from pydantic import BaseModel

class EmailInput(BaseModel):
    email: str

# Updated Prediction model for multi-label output
class MultiLabelPrediction(BaseModel):
    email_text: str
    predicted_intents: list[str]
    # Optionally, include probabilities for each predicted intent
    # probabilities: dict[str, float] # Example: {"Intent_A": 0.9, "Intent_C": 0.7}
