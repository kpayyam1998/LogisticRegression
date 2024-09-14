from src.utils.utils import evaluate_model
class ModelPrediction:
    def __init__(self,X_test,y_test):
        self.X_test=X_test
        self.y_test=y_test
        

    def prediction(self,model):
        return model.predict(self.X_test)


    def evaluation(self,y_pred):
        evaluate_model(y_pred,self.y_test)
    