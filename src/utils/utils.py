import os
import pickle 
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report
# from abc import ABC,abstractmethod

# class ModelEvaluation(ABC):
#     @abstractmethod
#     def evaluate(self, ActualValue, PredictedValue):
#         pass

def resultMetrix():
    return "Kp"

def load_dataset(file_path):
    # Check if dataset file exists in the data directory
    file_path = os.path.join(file_path, 'dataset.csv')
    if not os.path.exists(file_path):
        print(f"Dataset not found at {file_path}")
        return None
    df = pd.read_csv(file_path)
    return df


def save_model(model, model_name):
    # save the model in the models directory
    if not os.path.exists('./models'):
        os.makedirs('./models')
    model_path = os.path.join('./models', model_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def load_model(file_path):
    # Load the model from the models directory
    model_path = os.path.join(file_path,'logistic_regression_model.pkl')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def evaluate_model(predicted_value, actual_value):

    # Use sklearn's accuracy_score and classification_report for model evaluation
    accuracy = accuracy_score(predicted_value, actual_value)
    report = classification_report(predicted_value, predicted_value)

    if not  os.path.exists('./metrics'):
        os.makedirs('./metrics')
    # store the results in the metrics folder as txt files
    metrics_path = os.path.join('./metrics', 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(report)
    print(f"Model evaluation results saved to {metrics_path}")


