# your_project_name/src/config.py

from pathlib import Path

# --- Project & Data Paths ---
# This robustly finds the project root directory, which contains the 'src' folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# This points to the directory where you should place the unzipped 'food-101' folder.
# The structure should be: your_project_name/data/raw/food-101/
RAW_FOOD101_ROOT = RAW_DATA_DIR / "food-101"

# This is the directory where the prepare_dataset.py script will save its output.
PROCESSED_DATA_FOR_LORA = PROCESSED_DATA_DIR / "food_for_lora"

# --- Output Paths ---
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"      # For saving trained LoRA files
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "generated_images" # For saving your final images

# --- Model & Tokenizer IDs ---
# The base model for fine-tuning. Stable Diffusion 1.5 is a great standard for LoRA.
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# The model used for automatically generating captions for our dataset.
CAPTIONING_MODEL_ID = "Salesforce/blip-image-captioning-large"

# --- Data Preparation Parameters ---
# The unique "trigger word" that will activate your fine-tuned style.
TRIGGER_WORD = "pro_food_photo"

# The resolution of images for training.
IMAGE_RESOLUTION = (512, 512)

# Safety limit for testing the preparation script.
# Set to None to process all 75,750 training images (will take several hours).
# Set to a small number like 100 for a quick test run.
LIMIT = 100

# --- Fine-Tuning Parameters ---
# These will be used by your upcoming train.py script.
TRAIN_BATCH_SIZE = 1
MAX_TRAIN_STEPS = 3000   # Adjust based on the size of your processed dataset
LEARNING_RATE = 1e-4
CHECKPOINTING_STEPS = 500 # Save a checkpoint every N steps