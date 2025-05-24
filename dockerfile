FROM python:3.11-slim
WORKDIR /app

# Install uv (the Python package manager you are using)
RUN pip install uv

# Copy only the dependency definition file first
COPY pyproject.toml ./

# Install dependencies using uv
# This layer will be cached if pyproject.toml doesn't change,
# speeding up builds when only application code changes.
RUN uv sync --active

# Copy the application source code and necessary assets
COPY app.py ./
COPY src/ ./src/
COPY src/constants/label_map.json ./src/constants/



# The train.py script is generally not needed for the runtime image if the model
# is pre-trained and its outputs (model files, label_map.json) are copied above.
# If you have a specific reason to include it (e.g., for utilities or dynamic training),
# you can uncomment the line below.
# COPY train.py ./

EXPOSE 8000

# Command to run the FastAPI application using uvicorn, managed by uv.
# Ensure 'uvicorn' is listed as a project dependency in your pyproject.toml.
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]