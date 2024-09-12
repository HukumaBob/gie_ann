from pathlib import Path
from PIL import Image
import uuid

# Directories for raw and processed images
RAW_DATA_DIR = Path('raw_images')
PROCESSING_DATA_DIR = Path('processing_images')

# Output directories for different categories
OUTPUT_DIRS = {
    'esophagus': PROCESSING_DATA_DIR / 'esophagus',
    'stomach': PROCESSING_DATA_DIR / 'stomach',
    'duodenum': PROCESSING_DATA_DIR / 'duodenum'
}

# Target image format (e.g., 'PNG' or 'JPEG')
TARGET_FORMAT = 'PNG'
# Target resolution for the images
TARGET_SIZE = (224, 224)

# Function to process a single image
def process_image(input_path, output_path):
    try:
        with Image.open(input_path) as img:
            img = img.convert('RGB')  # Convert to 3-channel RGB
            img = img.resize(TARGET_SIZE)  # Resize to the target size
            img.save(output_path, format=TARGET_FORMAT)  # Save in the desired format
            print(f"Processed: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

# Main function to process all images
def preprocess_images(raw_data_dir, output_dirs):
    for category, output_dir in output_dirs.items():
        input_dir = raw_data_dir / category
        output_dir.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist
        
        for filename in input_dir.iterdir():
            if filename.is_file():
                # Generate a random UUID for anonymizing the file name
                anonymized_name = f"{uuid.uuid4()}.{TARGET_FORMAT.lower()}"
                output_path = output_dir / anonymized_name
                process_image(filename, output_path)

# Call the main function to start the preprocessing
preprocess_images(RAW_DATA_DIR, OUTPUT_DIRS)
