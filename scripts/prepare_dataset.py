# your_project_name/scripts/prepare_dataset.py

import os
import sys
from pathlib import Path
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

# --- Add the 'src' directory to the Python path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# ----------------------------------------------------

# Import all settings from our config file in the 'src' directory
from src import config

def main():
    """
    Main function to process the dataset based on settings in config.py.
    """
    print("--- Starting Dataset Preparation ---")
    print(f"Raw Data Source: {config.RAW_FOOD101_ROOT}")
    # --- CORRECTED LINE ---
    print(f"Processed Data Destination: {config.PROCESSED_DATA_FOR_LORA}")

    # Create the output directory if it doesn't exist
    # --- CORRECTED LINE ---
    config.PROCESSED_DATA_FOR_LORA.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load the AI model for captioning ---
    print(f"Loading captioning model: {config.CAPTIONING_MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(config.CAPTIONING_MODEL_ID)
    captioning_model = BlipForConditionalGeneration.from_pretrained(
        config.CAPTIONING_MODEL_ID, torch_dtype=torch.float16
    ).to(device)
    print(f"BLIP model loaded successfully to '{device}'.")

    # --- Step 2: Read the official training file list ---
    train_list_path = config.RAW_FOOD101_ROOT / "meta" / "train.txt"
    with open(train_list_path, 'r') as f:
        train_files = [line.strip() for line in f.readlines()]

    if config.LIMIT is not None:
        print(f"Processing a limited subset of {config.LIMIT} images for testing.")
        train_files = train_files[:config.LIMIT]
    else:
        print(f"Processing all {len(train_files)} training images.")

    # --- Step 3: Loop through, process, and save each image ---
    print("Processing images and generating captions...")
    for item_path in tqdm(train_files, desc="Processing Images"):
        try:
            class_name = item_path.split('/')[0].replace('_', ' ')
            image_path = config.RAW_FOOD101_ROOT / "images" / f"{item_path}.jpg"

            # Load and resize the image
            image = Image.open(image_path).convert("RGB").resize(config.IMAGE_RESOLUTION)

            # Generate caption with BLIP
            inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
            generated_ids = captioning_model.generate(**inputs, max_new_tokens=50)
            base_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            # Construct the final caption
            final_caption = f"{config.TRIGGER_WORD}, a professional photo of {class_name}, {base_caption}"

            # Save the processed image and caption
            output_filename_base = os.path.basename(item_path)
            # --- CORRECTED LINES ---
            image.save(config.PROCESSED_DATA_FOR_LORA.joinpath(f"{output_filename_base}.png"))
            with open(config.PROCESSED_DATA_FOR_LORA.joinpath(f"{output_filename_base}.txt"), 'w') as f:
                f.write(final_caption)

        except Exception as e:
            print(f"Skipping file {item_path} due to error: {e}")

    print("\n-----------------------------------------")
    print("✅ Dataset preparation complete!")
    # --- CORRECTED LINE ---
    print(f"Your training-ready data is located at: {config.PROCESSED_DATA_FOR_LORA}")

if __name__ == '__main__':
    main()