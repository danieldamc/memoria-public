import os
import gdown

# Script to download model weights from Google Drive
# using their file IDs.

FILE_ID_1 = "1YCZi6Oz6RLFiszWAuumkZ7EIZjpr8Y0E"
OUTPUT_NAME_1 = "segmentation_model.pt"

FILE_ID_2 = "1A2I7R70_GcOdeJmCp1LvC_OrP-zwG6NM"
OUTPUT_NAME_2 = "superresolution_model.pth"

FOLDER_PATH = "weights"

def download_file(file_id: str, output_name: str) -> None:
    """Download a file from Google Drive if it doesn't already exist.
    Args:
        file_id (str): The Google Drive file ID.
        output_name (str): The name of the output file.
    Returns:
        None
    """
    if not os.path.exists(output_name):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_name, quiet=False)
        return
    print(f"{output_name} file already exists. Skipping download.")
    return

def main() -> None:
    FILE_PATH_1 = os.path.join(FOLDER_PATH, OUTPUT_NAME_1)
    download_file(FILE_ID_1, FILE_PATH_1)

    FILE_PATH_2 = os.path.join(FOLDER_PATH, OUTPUT_NAME_2)
    download_file(FILE_ID_2, FILE_PATH_2)
    return 0

if __name__ == "__main__":
    main() 