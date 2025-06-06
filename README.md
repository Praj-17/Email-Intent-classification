# Email Intent Classification

This project implements a machine learning model to classify email intents. It uses a transformer-based model (BERT) fine-tuned for multi-label text classification to identify one or more intents from the body of an email. The application is served via a FastAPI backend with a simple HTML/JavaScript frontend.

## Project Links

*   **Live Application:** [https://email-classification-real-estate.onrender.com/](https://email-classification-real-estate.onrender.com/)
*   **GitHub Repository:** [https://github.com/Praj-17/Email-Intent-classification](https://github.com/Praj-17/Email-Intent-classification)
*   **DockerHub Repository:** [prajwal1717/email-classification-real-estate](https://hub.docker.com/r/prajwal1717/email-classification-real-estate)

## Features

*   Multi-label email intent classification.
*   RESTful API for predictions using FastAPI.
*   Interactive web interface for easy testing.
*   Logging for application monitoring and training.
*   Containerized with Docker for easy deployment.

## Technology Stack

*   **Backend:** Python, FastAPI, Uvicorn
*   **ML/NLP:** Transformers (Hugging Face), PyTorch, Scikit-learn
*   **Frontend:** HTML, CSS, JavaScript
*   **Data Handling:** Pandas, NumPy
*   **Logging:** Loguru
*   **Containerization:** Docker

## Project Structure

Email Intent classification/  
├── .github/workflows/  
├── .venv/ # Virtual environment  
├── archives/ # For storing older files or outputs  
├── data/ # CSV data for training (e.g., data.csv)  
├── logs/ # Application and training logs  
├── src/  
│ ├── constants/  
│ │ └── label_map.json # Generated by train.py, stores label mappings  
│ ├── data/ # Potentially for data-related scripts (if any)  
│ ├── outputs/ # Model checkpoints and outputs from training (Note: app.py loads from root 'outputs')  
│ │ └── checkpoint-xxx/ # Saved model and tokenizer from fine-tuning  
│ ├── templates/  
│ │ └── index.html # Frontend HTML  
│ ├── init.py  
│ ├── inference.py  
│ ├── models.py  
│ └── train.py  
├── .gitignore  
├── .python-version  
├── app.py  
├── dockerfile  
├── pyproject.toml  
├── README.md  
├── requirements.txt  
└── uv.lock # Lock file for uv package manage

*Note on structure: `train.py` saves model checkpoints to `PROJECT_ROOT/outputs/`. `app.py` loads the model from this location (e.g., `PROJECT_ROOT/outputs/checkpoint-320`).*

## Setup and Run

You can set up and run this project either locally using a Python virtual environment or using Docker.

### 1. Clone the Repository

First, clone the repository to your local machine:
```bash
git clone https://github.com/Praj-17/Email-Intent-classification.git
cd Email-Intent-classification
```

### 2. Option A: Local Setup and Execution

#### 2.1. Create a Virtual Environment and Install Dependencies
It's recommended to use a virtual environment (e.g., venv, conda).
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Or if using uv:
# uv pip install -r requirements.txt
```

#### 2.2. Data Preparation
*   Place your training data in `data/data.csv`.
*   The CSV should have at least two columns: one for the email text (default name "Email Body") and one for the intent categories (default name "Intent Category").
*   Multiple intents for a single email should be comma-separated in the "Intent Category" column (e.g., "Inquiry, Complaint").

#### 2.3. Run the Training Script
This will process the data, train the model, save the fine-tuned model to `outputs/checkpoint-xxx/`, and save `label_map.json` to `src/constants/`.
```bash
python src/train.py
```
After training, note the specific checkpoint directory (e.g., `outputs/checkpoint-320`) as `app.py` is configured to load from a specific checkpoint. You may need to update `model_dir_path` in `app.py` if the best checkpoint has a different number.

#### 2.4. Run the FastAPI Application
```bash
python app.py
```
Or using Uvicorn for more production-like settings:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
The application will be available at `http://localhost:8000`.

### 3. Option B: Docker Setup and Execution

The project includes a `dockerfile` to build a Docker image. This method is recommended for easier deployment and consistent environments.

**Prerequisites:**
*   Ensure you have Docker installed and running on your system.
*   Prepare your data as described in step `2.2. Data Preparation` above. The Docker build process will copy the `data` directory.
*   **Important:** If you have already trained a model locally and want to use it in Docker, ensure the `outputs` directory (containing your `checkpoint-xxx` and `label_map.json` in `src/constants`) is present. The Dockerfile copies these. If you haven't trained a model, the image will not contain a pre-trained model unless you modify the Dockerfile to run `train.py` during the build (not included by default).

#### 3.1. Build the Docker Image
```bash
docker build -t prajwal1717/email-classification-real-estate .
```

#### 3.2. Run the Docker Container
This command runs the container and maps port 8000 of the container to port 8000 on your host machine.
```bash
docker run -p 8000:8000 prajwal1717/email-classification-real-estate
```
Access the application at `http://localhost:8000`.

#### 3.3. Push to DockerHub (Optional)
If you want to share your image on DockerHub:
```bash
docker login
docker push prajwal1717/email-classification-real-estate
```

## Data Collection and Preparation (`src/train.py`)

1.  **Loading Data:** The script loads data from `data/data.csv` using Pandas. It expects columns named "Email Body" (renamed to `text`) and "Intent Category" (renamed to `label_str`).
2.  **Label Processing:**
    *   The "Intent Category" can contain comma-separated strings for multi-label classification. These are converted into a list of strings (`label_list`).
    *   A `label_map.json` file is created and saved to `src/constants/label_map.json`. This file maps each unique label string to an integer ID (`label2id`) and vice-versa (`id2label`), and also stores an ordered list of all unique labels (`labels_ordered`). This file is crucial for the model and the application.
3.  **Multi-Label Binarization:**
    *   `MultiLabelBinarizer` from `scikit-learn` is used to convert the `label_list` for each email into a multi-hot encoded vector. This vector has a length equal to the total number of unique labels, with `1` at indices corresponding to the labels present in the email and `0` otherwise.
    *   These encoded labels are stored as `np.float32` arrays, required for training with `BCEWithLogitsLoss`.
4.  **Train-Test Split:** The dataset is split into training and validation sets using `train_test_split`.

## Training Process (`src/train.py`)

1.  **Tokenizer and Model Initialization:**
    *   A pre-trained tokenizer (e.g., "bert-base-uncased") and sequence classification model are loaded.
    *   `problem_type="multi_label_classification"` is specified.
    *   `num_labels`, `label2id`, and `id2label` are configured for the model.

2.  **Tokenization:**
    *   Email text is tokenized, padded, and truncated. Data is converted to Hugging Face `Dataset` objects and formatted for PyTorch.

3.  **Training Arguments:**
    *   `TrainingArguments` are configured, including output directory (`./outputs`), evaluation strategy, learning rate, batch sizes, epochs, and metric for selecting the best model (`f1_macro`). `load_best_model_at_end=True` ensures the best checkpoint is retained.

4.  **Compute Metrics:**
    *   A `compute_metrics` function calculates F1-scores (macro, micro, samples) and subset accuracy for multi-label evaluation. Logits are converted to probabilities via sigmoid, then thresholded.

5.  **Trainer Initialization and Training:**
    *   A `Trainer` object is initialized and `trainer.train()` starts the fine-tuning.

6.  **Saving Model:**
    *   The Hugging Face `Trainer` automatically saves the best model checkpoint (based on `metric_for_best_model`) to a subdirectory within `output_dir` (e.g., `./outputs/checkpoint-xxx/`). This directory contains the model weights, configuration, and tokenizer files. `app.py` is configured to load from such a checkpoint directory.

7.  **Evaluation Report:**
    *   After training, `trainer.predict()` is run on the validation set. A `classification_report` and per-label confusion matrices are logged.

## API Endpoints (`app.py`)

*   `GET /`: Serves the `index.html` frontend.
*   `POST /predict`:
    *   Accepts JSON: `{"email": "email text here"}`.
    *   Returns JSON:
        *   Success: `{"predicted_intents": ["Intent1", "Intent2"], "confidence_scores": [0.9, 0.8]}` (Note: `confidence_scores` are illustrative; actual implementation details in `src/inference.py`).
        *   No intent: `{"predicted_intents": [], "confidence_scores": []}`.
        *   Error: `{"predicted_intents": ["error during prediction"], "error_detail": "Error message"}`.
*   `GET /healthz`: Health check, returns `{"status": "ok"}`.

## Docker Deployment

The project includes a `dockerfile` to build a Docker image. See the "Setup and Run > Option B: Docker Setup and Execution" section for detailed instructions.

## Future Improvements

*   More robust multi-label stratification during train/test split.
*   Hyperparameter tuning.
*   Displaying confidence scores per intent on the frontend.
*   Unit and integration tests.

## Contact

For any inquiries or feedback, please reach out:
*   **Email:** [pwaykos1@gmail.com](mailto:pwaykos1@gmail.com)
*   **LinkedIn / Resume:** [View Resume](https://drive.google.com/file/d/1OiSCu4e_1R7cawKSU80cr63Cd2-4OVq7/view?usp=drivesdk)
*   **GitHub:** [Praj-17](https://github.com/praj-17)
