from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from loguru import logger
import sys
import os
from src.inference import perform_email_prediction

# Assuming src.models contains EmailInput and MultiLabelPrediction
# This requires 'src' to be a package or the project root to be in PYTHONPATH
from src.models import EmailInput, MultiLabelPrediction

# --- Global Configuration & Setup ---
APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_logging():
    """Configures Loguru for application logging."""
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    
    logs_dir = os.path.join(APP_BASE_DIR, "logs")
    if not os.path.exists(logs_dir):
        try:
            os.makedirs(logs_dir)
            logger.info(f"Successfully created logs directory: {logs_dir}")
        except OSError as e:
                logger.error(f"Could not create logs directory: {e}")
            # Depending on policy, might exit or continue with stderr logging only
    
    logger.add(os.path.join(logs_dir, "app.log"), rotation="500 MB", level="INFO")
    logger.info("Logging configured.")

def load_app_config():
    """Loads application configurations like paths and label mappings."""
    logger.info("Loading application configuration...")

    # Templates directory (as per your original structure)
    templates_dir = os.path.join(APP_BASE_DIR, "src", "templates")
    if not os.path.isdir(templates_dir):
        logger.warning(f"Templates directory not found at {templates_dir}. HTML frontend might not work.")
        # Optionally create it or handle as needed
        # For now, we'll let Jinja2Templates handle it later if still missing.

    # Corrected path for label_map.json (train.py saves it to project root)
    label_map_path = os.path.join(APP_BASE_DIR, "src","constants", 'label_map.json')
    if not os.path.exists(label_map_path):
        logger.error(f"label_map.json not found at {label_map_path}. Ensure train.py has run and saved it.")
        sys.exit(f"Error: label_map.json not found at {label_map_path}")

    with open(label_map_path) as f:
        label_map_content = json.load(f)

    if "labels_ordered" not in label_map_content:
        logger.error("'labels_ordered' key not found in label_map.json. This is required.")
        sys.exit("Error: 'labels_ordered' not found in label_map.json.")
    
    all_labels_ordered = label_map_content["labels_ordered"]
    num_labels = len(all_labels_ordered)
    logger.info(f"Label mapping loaded. {num_labels} labels: {all_labels_ordered}")

    return templates_dir, all_labels_ordered, num_labels

def load_model_and_tokenizer(num_labels_from_config: int):
    """Loads the ML model and tokenizer."""
    logger.info("Loading model and tokenizer...")

    # Corrected model path: train.py saves to <project_root>/outputs/checkpoint-xxx
    # APP_BASE_DIR is the directory of app.py (project root)
    model_dir_path = os.path.join(APP_BASE_DIR,"src", 'outputs', 'checkpoint-320') # Removed 'src' from this path
    if not os.path.isdir(model_dir_path):
        logger.error(f"Model directory not found at {model_dir_path}. Ensure train.py has run and saved the model to ./outputs/checkpoint-xxx/.")
        sys.exit(f"Error: Model directory not found at {model_dir_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir_path,
        problem_type="multi_label_classification",
        num_labels=num_labels_from_config
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir_path)
    logger.info("Model and tokenizer loaded successfully.")

    # Sanity check
    if model.config.num_labels != num_labels_from_config:
        logger.error(
            f"CRITICAL MISMATCH: Model config num_labels ({model.config.num_labels}) "
            f"does not match label_map num_labels ({num_labels_from_config}). "
            "This will lead to incorrect predictions."
        )
        sys.exit("Error: Model and label_map num_labels mismatch.")
    logger.info(f"Model configured num_labels ({model.config.num_labels}) matches label_map ({num_labels_from_config}).")
    
    return model, tokenizer, device



# --- Initialize Application Components ---
setup_logging()
TEMPLATES_DIR, ALL_LABELS_ORDERED, NUM_LABELS = load_app_config()
templates = Jinja2Templates(directory=TEMPLATES_DIR) # Initialize after TEMPLATES_DIR is confirmed/created
MODEL, TOKENIZER, DEVICE = load_model_and_tokenizer(NUM_LABELS)

PREDICTION_THRESHOLD = 0.3 # Can be tuned or moved to a config file

# --- FastAPI Application ---
app = FastAPI(title="Email Intent Classifier")
logger.info("FastAPI app initialized.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logger.info("Root HTML page requested.")
    index_html_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_html_path):
        logger.error(f"index.html not found in templates directory: {index_html_path}")
        # Consider providing a fallback or a more user-friendly error page
        return HTMLResponse(content="<html><body><h1>Error: index.html not found</h1><p>Please ensure an index.html file exists in the '{TEMPLATES_DIR}' directory.</p></body></html>", status_code=500)
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/healthz')
def healthz():
    logger.info("Health check endpoint called.")
    return {'status': 'ok'}

@app.post('/predict', response_model=MultiLabelPrediction)
def predict_intent_endpoint(input_data: EmailInput):
    prediction_result = perform_email_prediction(
        email_text=input_data.email,
        model=MODEL,
        tokenizer=TOKENIZER,
        device=DEVICE,
        all_labels=ALL_LABELS_ORDERED,
        pred_threshold=PREDICTION_THRESHOLD
    )
    # If perform_email_prediction now returns a dict that might include an error,
    # you might want to check for that and raise HTTPException if needed,
    # or ensure MultiLabelPrediction can handle the error structure.
    # For now, assuming MultiLabelPrediction matches the successful output structure.
    if "error_detail" in prediction_result:
         # MultiLabelPrediction expects 'predicted_intents' to be a list of strings.
         # If there's an error, 'predicted_intents' is ["error during prediction"].
         # We could raise an HTTPException here for a proper API error response.
         # raise HTTPException(status_code=500, detail=f"Prediction error: {prediction_result['error_detail']}")
         # For simplicity, and if client handles this, we can return as is:
         pass # The structure with "error during prediction" matches existing error flow

    return prediction_result

if __name__ == "__main__":
    import uvicorn
    # This allows running directly with `python app.py`
    # Ensure uvicorn is installed: pip install uvicorn
    logger.info("Starting Uvicorn server for app.py...")
    # The app object is 'app', and app.py is the file.
    # Host 0.0.0.0 makes it accessible on the network.
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)