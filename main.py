from src.data.data_loader import DataLoader
from src.data.data_loader import DatasetConfig
from src.utils.utils import save_model
from src.models.model_prediction import ModelPrediction
from src.models.model_training import model_training


class Pipeline:
    def __init__(self,data_path):
        self.data_path = DatasetConfig(data_path)
        
    def run(self):
        dl_obj=DataLoader(self.data_path)
        df=dl_obj.LoadDataset()

        # train model
        train_obj=model_training(df)
        X,y=train_obj.split_data()
        X_train, X_test, y_train, y_test=train_obj.train_test_split(X, y)
        model=train_obj.logistic_regression(X_train, y_train)

        # prediction and evaluation
        predobj=ModelPrediction(X_test,y_test)
        y_pred=predobj.prediction(model)
        predobj.evaluation(y_pred)

        print("Model Done")

if __name__ == '__main__':
    # Specify the data directory
    data_dir = './data/raw'

    # Create a pipeline instance
    pipeline = Pipeline(data_dir)

    # Run the pipeline
    pipeline.run()