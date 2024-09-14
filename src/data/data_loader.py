import os
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from dataclasses import dataclass


from src.utils.utils import load_dataset
@dataclass
class DatasetConfig:
    data_dir: str 

class DatasetGeneration:
    def __init__(self, n_samples=10000, n_features=12, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, shuffle=True, random_state=None):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.n_repeated = n_repeated
        self.n_classes = n_classes
        self.n_clusters_per_class = n_clusters_per_class

    def generate_dataset(self):
        X, y = make_classification(n_samples=self.n_samples, n_features=self.n_features, n_informative=self.n_informative, n_redundant=self.n_redundant, n_repeated=self.n_repeated, n_classes=self.n_classes, n_clusters_per_class=self.n_clusters_per_class, weights=None, shuffle=True, random_state=None)
        df=pd.DataFrame(X)
        df['target']=y
        return df
    

    def save_dataset(self):
        df = self.generate_dataset()
        # Check if data directory exists or not
        data_dir = '../../data/raw'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        # Save dataset to csv file in the data directory
        file_path = os.path.join(data_dir, 'dataset.csv')
        if os.path.exists(file_path):
            os.remove(file_path)  # Remove existing file if it exists
        # Save dataset to csv file
        df.to_csv(file_path, index=False)
        print(f"Dataset saved to {file_path}")
       
class DataLoader:
    def __init__(self, config: DatasetConfig):
        self.file_path = config.data_dir
    
    def LoadDataset(self):
        df = load_dataset(self.file_path)
        return df


if __name__=="__main__":
    ds=DatasetConfig('../../data/raw')
    #ds_obj=DataLoder(ds)
    #print(ds_obj.LoadDataset())
        