import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

def main():
	features = pd.read_excel("Imaging_Features.xlsx")
	features = features.dropna()
	labels = pd.read_csv("case_metadata.csv")

	df = features.merge(labels, on='Patient ID',how='inner')
	features = df.drop(columns=['Patient ID', 'ER','HER2','PR'])

	er_labels = df['ER']
	her_labels = df['HER2']
	pr_labels = df['PR']

	# Define a pipeline to search for the best combination of PCA truncation
	# and classifier regularization.
	pca = PCA()
	# set the tolerance to a large value to make the example faster
	logistic = LogisticRegression(max_iter=10000, tol=0.1)
	pipe = Pipeline(steps=[('scaler', StandardScaler()), ('pca', pca), ('logistic', logistic)])
	param_grid = {
    	'pca__n_components': [5, 10, 50, 100],
    	'logistic__C': np.logspace(-4, 4, 4),
	}
	search = GridSearchCV(pipe, param_grid, n_jobs=-1)
	search.fit(features, er_labels)
	print("Best parameter (CV score=%0.3f):" % search.best_score_)
	print(search.best_params_)
	
	pca.fit(features)

	fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
	ax0.plot(np.arange(1, pca.n_components_ + 1),
	         pca.explained_variance_ratio_, '+', linewidth=2)
	ax0.set_ylabel('PCA explained variance ratio')

	ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
	            linestyle=':', label='n_components chosen')
	ax0.legend(prop=dict(size=12))

	# For each number of components, find the best classifier results
	results = pd.DataFrame(search.cv_results_)
	components_col = 'param_pca__n_components'
	best_clfs = results.groupby(components_col).apply(
	    lambda g: g.nlargest(1, 'mean_test_score'))

	best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
	               legend=False, ax=ax1)
	ax1.set_ylabel('Classification accuracy (val)')
	ax1.set_xlabel('n_components')

	plt.xlim(-1, 100)

	plt.tight_layout()
	plt.savefig("PCA.png")

	pca = PCA(n_components=10)
	features = pca.fit_transform(features)
	features_train, features_test, er_train, er_test = train_test_split(features, er_labels, test_size=0.20, random_state=42)
	mod_features = features_train
	logistic.fit(mod_features,er_train)

	metrics.plot_roc_curve(logistic, features_test, er_test)
	plt.plot([0,1],[0,1],'-')
	plt.savefig("AUC")                                  

if __name__ == '__main__':
	main()