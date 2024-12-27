from PIL import Image
import pillow_heif

# Register HEIF support
pillow_heif.register_heif_opener()

def convert_heic_to_jpeg(input_path, output_path):
    try:
        # Open the HEIC file
        with Image.open(input_path) as img:
            # Convert to RGB mode if not already in RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Save as JPEG
            img.save(output_path, "JPEG")
        print(f"Converted: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error converting {input_path}: {e}")

import os

def batch_convert_heic_to_jpeg(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith('.heic'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name.rsplit('.', 1)[0] + '.jpg')
            convert_heic_to_jpeg(input_path, output_path)

# Example usage
batch_convert_heic_to_jpeg("../../resources/evaluation_dataset/Zosia_RGB_HIEC", "../../resources/evaluation_dataset/Zosia_RGB_JPG")

