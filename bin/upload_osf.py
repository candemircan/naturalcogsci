import os
from osfclient.api import OSF
from naturalcogsci.helpers import get_project_root
from tqdm import tqdm

# Configuration
PROJECT_ROOT = get_project_root()
PROJECT_ID = 'h3t52' 
AUTH_TOKEN = os.getenv('OSF_TOKEN')  # Read the OSF authentication token from an environment variable
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')  # Directory containing folders to upload

def upload_folder(osf, project, folder_path):
    folder_name = os.path.basename(folder_path)
    storage = project.storage('osfstorage')
    
    # Check if the folder already exists on OSF
    remote_folder = None
    for folder in storage.folders:
        if folder.name == folder_name:
            remote_folder = folder
            break
    
    # If the folder doesn't exist, create it
    if not remote_folder:
        remote_folder = storage.create_folder(folder_name)
    
    # Get list of existing files in the remote folder
    existing_files = {file.name for file in storage.files if file.path.startswith(f'/{folder_name}/')}

    for root, dirs, files in tqdm(os.walk(folder_path)):
        for file in files:

            try:
                file_path = os.path.join(root, file)
                remote_path = os.path.relpath(file_path, folder_path)
                if file not in existing_files:
                    print(f"Uploading file: {file}")
                    with open(file_path, 'rb') as fp:
                        storage.create_file(f'/{folder_name}/{remote_path}', fp, update=False)
                else:
                    print(f"Skipping existing file: {file}")
            except FileExistsError:
                pass

def main():
    if not AUTH_TOKEN:
        raise ValueError("OSF_AUTH_TOKEN environment variable not set")

    osf = OSF(token=AUTH_TOKEN)
    project = osf.project(PROJECT_ID)

    for folder_name in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            upload_folder(osf, project, folder_path)

if __name__ == "__main__":
    main()