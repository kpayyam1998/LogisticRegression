import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.data.data_loader import DataLoader
from src.data.data_loader import DatasetConfig
from src.utils.utils import save_model
from src.models.model_prediction import ModelPrediction

class model_training:
    def __init__(self,df:pd.DataFrame) -> None:
        self.df = df
    
    def split_data(self):

        # Split data into features (X) and target (y)
        X = self.df.drop('target', axis=1)
        y = self.df['target']

        return X,y
    
    def train_test_split(self, X, y, test_size:float =0.2):
         # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        return X_train, X_test, y_train, y_test
    
    def logistic_regression(self, X_train, y_train):
        # Train a logistic regression model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # save logistic regression model
        save_model(model, 'logistic_regression_model.pkl')

        return model
    

if __name__=='__main__':
    ds=DatasetConfig('../../data/raw')
    ds_obj=DataLoader(ds)
    df=ds_obj.LoadDataset()
    train_obj=model_training(df)
    X,y=train_obj.split_data()
    X_train, X_test, y_train, y_test=train_obj.train_test_split(X, y)
    model=train_obj.logistic_regression(X_train, y_train)

    predobj=ModelPrediction(X_test,y_test)
    y_pred=predobj.prediction(model)
    predobj.evaluation(y_pred)

    print("Model Done")