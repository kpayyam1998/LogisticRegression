import os 
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO,format='[%(asctime)s]:%(message)s')
list_of_file=[
    "config/config.yml",
    "data/raw/sample.txt",
    "data/processed/sample.txt",
    "notebook/EDA.ipynb",
    "notebook/data_preprocess.ipynb",
    "notebook/feature_selection.ipynb",
    "notebook/model_buildings.ipynb",
    "src/data/_init_.py",
    "src/data/data_loader.py",
    "src/data/data_preprocess.py",
    "src/models/_init_.py",
    "src/models/model_selector.py",
    "src/models/model_training.py",
    "src/models/model_prediction.py",
    "src/features/_init_.py",
    "src/features/feature_engineering.py",
    "src/utils/_init_.py",
    "src/utils/utils.py",
    "models/models.txt",
    "srcipts/_init_.py",
    "scripts/model_train.py",
    "scripts/test_prediction.py",
    "deployment/model.txt",
    "flask_api.py",
    ".gitignore",
    "README.md",
    "requirements.txt",
    "setup.py"
]
for files in list_of_file:
    file_path=Path(files)
    file_dir,file_name=os.path.split(files)
    if file_dir!="":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating file directory:{file_dir} for the file :{file_name}")
    if (not os.path.exists(file_path)) or(os.path.getsize(file_path)==0):
        with open(file_path,'w') as f:
            pass
            logging.info(f"Creating empty file:{file_path}")
    else:
        logging.info(f"{file_name} is already created")