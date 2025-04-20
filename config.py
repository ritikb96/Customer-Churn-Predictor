import sys
import os

# Get the root directory (project root)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add root to sys.path
sys.path.append(root_dir)
from utils.Handytoolbox import *
import mlflow
mlflow.set_experiment("Churn Prediction")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
from steps.clean_data import clean_df
from steps.ingestdata import ingest_df
import mlflow.sklearn
import mlflow.xgboost



df = ingest_df(data_path='data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
(X_train,X_test,y_train,y_test) = clean_df(df)
models = [
    (
        "Logistic Regression", 
        LogisticRegression(C=1, solver='liblinear'), 
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "Random Forest", 
        RandomForestClassifier(class_weight= 'balanced', max_depth= 10, min_samples_leaf =1, min_samples_split= 17, n_estimators =149), 
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "XGBClassifier",
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'), 
        (X_train, y_train),
        (X_test, y_test)
    )

]


def generate_reports(models):
    best_model = None
    best_accuracy = 0
    best_report = None
    reports = []
    
    for model_name, model, (x_train, y_train), (x_test, y_test) in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        reports.append((model_name, report))
        
    
    return reports

results = generate_reports(models)
print(results)

for i,element in enumerate(models):
    model_name = element[0]
    model = element[1]
    _,report = results[i]

    with mlflow.start_run(run_name = model_name):
        mlflow.log_param('model', model_name)
        mlflow.log_metric('accuracy', report['accuracy'])
        mlflow.log_metric('recall_class_1', report['1']['recall'])
        mlflow.log_metric('recall_class_0', report['0']['recall'])
        mlflow.log_metric('f1_score_macro', report['macro avg']['f1-score']) 

        if "XGB" in model_name:
            mlflow.xgboost.log_model(model,"model")
        else:
            mlflow.sklearn.log_model(model,"model")






