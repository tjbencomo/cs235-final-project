import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def evaluate_model_pca(features, labels, name, gene):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42)
    model = RandomForestClassifier()
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA()), ('rf', model)])
    param_grid = {
    'pca__n_components': [5, 15, 25, 50, 100, 200],
    'rf__n_estimators' : [10, 50, 100, 200, 500],
    'rf__max_depth' : [10, 50, 100, 500, None]
    }
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    print(f"Starting grid search for {name}")
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    best_model = search.best_estimator_
    print("Plotting AUC")
    metrics.plot_roc_curve(best_model, X_test, y_test)
    plt.plot([0,1],[0,1],'-')
    plt.title(f"{gene} ROC")
    plt.savefig(f"{name}_AUC.png")


def evaluate_model(features, labels, name, gene):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42)
    model = RandomForestClassifier()
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('rf', model)])
    param_grid = {
    'rf__n_estimators' : [10, 50, 100, 200, 500],
    'rf__max_depth' : [10, 50, 100, 500, None]
    }
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    print(f"Starting grid search for {name}")
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    best_model = search.best_estimator_
    print("Plotting AUC")
    metrics.plot_roc_curve(best_model, X_test, y_test)
    plt.plot([0,1],[0,1],'-')
    plt.title(f"{gene} ROC")
    plt.savefig(f"{name}_AUC.png")

def main():
    features = pd.read_excel("Imaging_Features.xlsx")
    features = features.dropna()
    labels = pd.read_csv("case_metadata.csv")

    df = features.merge(labels, on='Patient ID',how='inner')
    features = df.drop(columns=['Patient ID', 'ER','HER2','PR'])

    er_labels = df['ER']
    her2_labels = df['HER2']
    pr_labels = df['PR']

    print("Evaluating Random Forest Models")
    evaluate_model(features, er_labels, "ER_Classifier", "ER")
    evaluate_model(features, pr_labels, "PR_Classifier", "PR")
    evaluate_model(features, her2_labels, "HER2_Classifier", "HER2")

    print("Evaluating PCA based Random Forest Models")
    evaluate_model_pca(features, er_labels, "ER_PCA_Classifier", "ER")
    evaluate_model_pca(features, pr_labels, "PR_PCA_Classifier", "PR")
    evaluate_model_pca(features, her2_labels, "HER2_PCA_Classifier", "HER2")


if __name__ == '__main__':
        main()
