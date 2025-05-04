#! /usr/bin/python

# from distutils.core import setup
'''
# setup(name='sktools',
 		version='1.0.5',
		date_updated = 5 Nov 2021 
 		py_modules=['sktools'])
'''
#basics	 
import numpy as np
import pandas as pd
import joblib
import os

#viz
import matplotlib.pyplot as plt
import seaborn as sns

#ML packages
from sklearn import model_selection
from sklearn import feature_selection
import sklearn.metrics as metrics

from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions

a = 0

b = a + 1


#### --- Feature
class modelling:
	def __init__(self):
		pass
	
	#Creates a dataframe for results comparison
	def results(models, cols=None):
		if cols==None:
			cols = ['Model','accuracy','train_score','test_score','train_roc_auc','test_roc_auc']	
		whole_set = []

		for key in models.keys():
			row=[key]
			for col in cols[1:]:
				row.append(models[key][col])
			whole_set.append(row)

		df = pd.DataFrame(whole_set, columns=cols)	
		return df

	#Single Set Modeler (X_train + y_train | X_test + y_test)
	def mono_modeler(model,X,y,fitted=None,verbose=False, out_save = False, plot_save=False, single_save=False,s_prefix = '',
		figsize=[18,8], font_scale=1, multi_class=False, ext='png', coef=False, sub_folder_level=0, in_save=False,
		 plotter=True, hspace=0.4, titlesize=20, pickle=False):
		'''Auto modelling tool for modelling either a train or test set. 
		For validation simply set the 'fitted' flag to True'''

		#if any save points is enables, check /directory
		if out_save==True or plot_save==True or single_save==True or pickle==True:
			utils.dir_check(s_prefix)

		if verbose: print('Fitting...')
		if fitted==None:
			fitted=model.fit(X,y)
		else:
			fitted=model	

		score = fitted.score(X,y)

		#obtain coefficients 
		if coef==True:	
			coef = fitted.coef_
			print('Coefficients mean: ', np.mean(coef))
		else:
			coef = 0

		#create predictions and CM
		if verbose: print('Creating predictions and CM...')
		preds = fitted.predict(X)
		confusion_matrix = metrics.confusion_matrix(y, preds)
		accuracy = metrics.accuracy_score(preds, y)

		#obtain fpr, tpr, score and plot ROC curve with train + test data	
		if multi_class==False:
			if verbose: print('Obtaining FPR, TPR, Predictions Probabilities and ROC...')
			#calc
			fpr,tpr, prob_preds, roc_auc = modelling.get_roc(fitted,X, y,mono=False).values()	
			ncols=2			
		else: #multiclass
			if verbose: print('Multiclass Classifier. will not compute ROC')
			prob_preds = fitted.predict_proba(X)
			fpr, tpr, roc_auc = (0,0,0)

			#prep variables for plotting
			ncols=1
			titlesize *=.8

		if plotter == True:
			fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)
			fig.suptitle(s_prefix.split('/')[sub_folder_level]+str(model), size=titlesize)
			sns.set_theme(font_scale=font_scale)
			sns.set_context('notebook', font_scale=font_scale)
			if multi_class == False:
				plotting.confusion_matrix(fitted,confusion_matrix, figsize=figsize, save=single_save, s_prefix=s_prefix, ax=axes[0])
				plotting.roc(fitted,fpr, tpr, roc_auc, figsize=figsize, save=plot_save, s_prefix=s_prefix, ax=axes[1])
				plt.subplots_adjust(hspace=hspace)
			else:				
				plotting.confusion_matrix(fitted,confusion_matrix, figsize=figsize, save=single_save, s_prefix=s_prefix, ax=axes)			
			
			if plot_save== True:	
				if verbose: print('Plotting')
				name = s_prefix + str(model).split('(')[0] + '.' + ext
				plt.savefig(name)

		#defining output
		output = {
			'fitted': fitted,
			'CM': confusion_matrix,
			'fpr': fpr,        
			'tpr': tpr,
			'score': score,
			'coef': coef,
			'preds': preds,
			'prob_preds':prob_preds,
			'roc_auc': roc_auc,
			'accuracy': accuracy
			}

		#generates pickle
		if pickle:
			name = s_prefix + str(model).split('(')[0] +'.joblib'
			joblib.dump(fitted, open(name,'wb'))	

		#saves input data
		if in_save:
			filename = s_prefix + str(model).split('(')[0]
			df_like = ['X','y']
			dfs = dict((k, output[k]) for k in df_like)
			for p in dfs.keys():
				pd.DataFrame(dfs[p]).to_csv(filename + '_' + p + '.csv')
			del dfs	
		
		#saving output
		if out_save:
			if verbose: print('Saving...')
			filename = s_prefix + str(model).split('(')[0]
			file = open(filename + '.txt','w')
			
			non_df = dict((k, output[k]) for k in['score','test_score','roc_auc','roc_auc', 'accuracy'])
			file.write(repr(non_df))
			file.close()
			del non_df
			
			df_like = ['X','y','CM','fpr','tpr','preds','prob_preds','tpr',\
			'preds','prob_preds','coef']
			if coef == False: 
				df_like.remove('coef')

			dfs = dict((k, output[k]) for k in df_like)
			for p in dfs.keys():
				pd.DataFrame(dfs[p]).to_csv(filename + '_' + p + '.csv')
			del dfs	

		#end and return
		if verbose: print('Finished!')
		return output	

	#Dual single set, parallel feed
	def dual_mono(name,model,X_train,y_train,X_test,y_test,train_params=None, test_params=None,
		figsize=[12,8], font_scale=1, multi_class=False, ext='png', coef=False, sub_folder_level=0,
		 hspace=0.4, titlesize=20):

		param_set = {'verbose':False, 'out_save':False, 'plot_save':False, 'single_save':False,'s_prefix':'',
			'plotter':True, 'pickle':False, 'font_scale':font_scale,'multi_class':multi_class,'ext':ext,'coef':coef,'sub_folder_level':sub_folder_level,
			'hspace':hspace,'titlesize':titlesize}	
		
		#Training
		if train_params==None:#default params
			train_params = param_set
		else:
			param_set.update(train_params)	
			train_params = param_set

		set_='train'
		train_params['s_prefix']=f'output/VAD/{name}/{set_}/'
		train_set=modelling.mono_modeler(model,X_train,y_train,**train_params)

		#Testing
		if test_params==None:
			test_params = param_set
		else:
			param_set.update(test_params)	
			test_params = param_set
			
		set_='test'
		test_params['s_prefix']=f'output/VAD/{name}/{set_}/'
		utils.dir_check(test_params['s_prefix'])
		test_set = modelling.mono_modeler(train_set['fitted'],X_test,y_test,**test_params)

		return {'name':name,'train':train_set,'test':test_set}
	
	#Joint Set Modeler (Auto-split--> X,y)
	def stereo_modeler(model,X,y, verbose=False, out_save = False, pickle=False, plot_save=False, s_prefix='',
		figsize=[15,8], font_scale=1,  test_size=0.2, random_state=20, super_verbose=False, comp=False,
		multi_class=False, ext='png', coef=False, rfe=False, sub_folder_level=0, plotter=True, hspace=0.4,
		titlesize=14, context='notebook'):
		'''Auto modelling tool'''
		#Split and Comparison
		if comp:
			X, X_train, X_test, y_train ,y_test, test_score, selector = modelling.feature_selector_comp(model,X,y, super_verbose=super_verbose, rfe=rfe)
			#fitting the model with the resulting set of features
			fitted = model.fit(X_train,y_train)
		else:
			X_train, X_test, y_train ,y_test = model_selection.train_test_split(X,y, test_size = test_size, random_state=random_state)
			#fitting the model with the resulting set of features
			if verbose==True: print('Fitting...')
			fitted = model.fit(X_train,y_train)
			#getting scores
			train_score = fitted.score(X_train,y_train)
			test_score = fitted.score(X_test,y_test)

		#obtain coefficients 
		if coef==True:	
			coef = fitted.coef_
			print('Coefficients mean: ', np.mean(coef))
		else:
			coef = 0

		#prepping Plotting space
		if plotter==True:
			fig, axes = plt.subplots(2,2, figsize=figsize)
			fig.suptitle(s_prefix.split('/')[sub_folder_level]+str(model), size=titlesize)
			sns.set_theme(font_scale = font_scale)
			sns.set_context(context, font_scale = font_scale)

		#Confusion Matrix with train and test data
		#create predictions and CM
		if verbose==True: print('Creating predictions and CM...')
		train_preds = fitted.predict(X_train)
		train_confusion_matrix = metrics.confusion_matrix(y_train, train_preds)
		
		test_preds=fitted.predict(X_test)
		test_confusion_matrix = metrics.confusion_matrix(y_test, test_preds)

		if plotter==True:		
			plotting.confusion_matrix(fitted,train_confusion_matrix, figsize=figsize, save=plot_save, s_prefix = s_prefix, ax=axes[0,0])
			plotting.confusion_matrix(fitted,test_confusion_matrix, figsize=figsize, save=plot_save, s_prefix = s_prefix, ax=axes[1,0])
			#axes[0,2] = plot_learning_curves(X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy(), fitted, suppress_plot=True, print_model = False, style = 'default')
			#plot_learning_curves(X_train, y_train, X_test, y_test, fitted, print_model = False, style = 'ggplot')

		#quick accuracy
		accuracy = metrics.accuracy_score(test_preds, y_test)

		#obtain fpr, tpr, score and plot ROC curve with train + test data
		if verbose==True: print('Obtaining FPR, TPR, Predictions Probabilities and ROC...')
		if multi_class==False :#obtain predictions, confusion_matrix and plot it if the model is NOT a multi-class model
		#train Data
			#calc
			train_fpr,train_tpr, train_prob_preds, train_roc_auc = modelling.get_roc(fitted,X_train, y_train).values()
			#plotting
			if plotter==True:
				plotting.roc(fitted,train_fpr, train_tpr, train_roc_auc,\
				 figsize=figsize, save=plot_save, s_prefix = s_prefix, ax=axes[0,1])
		#test Data
			#calc
			test_fpr,test_tpr, test_prob_preds, test_roc_auc = modelling.get_roc(fitted,X_test, y_test).values()
			#plotting
			if plotter==True:
				plotting.roc(fitted,test_fpr, test_tpr, test_roc_auc,\
				 figsize=figsize, save=plot_save, s_prefix = s_prefix, ax=axes[1,1])
		else: #multiclass
			print('multiclass')
			axes[0,1].set_visible(False)
			test_prob_preds = fitted.predict_proba(X_test)[:,1]
			fpr, tpr, _ = metrics.roc_curve(y_test, test_prob_preds)
		
		if plotter == True:
			plt.subplots_adjust(hspace=hspace)

			if plot_save== True:
				if verbose==True: print('Plotting')
				name = s_prefix + str(model).split('(')[0] + '.' + ext
				plt.savefig(name)

		if pickle:
			name = s_prefix + str(model).split('(')[0] +'.joblib'
			joblib.dump(fitted, open(name,'wb'))

		#defining output
		output = {
			'fitted': fitted,
			'X_train': X_train,
			'X_test': X_test,
		 	'y_train': y_train,
			'y_test': y_test,
			'train_CM': train_confusion_matrix,
			'test_CM': test_confusion_matrix,
			'train_fpr': train_fpr,
			'test_fpr': test_fpr,
			'train_tpr': train_tpr,
			'test_tpr':test_tpr,
			'train_score': train_score,
			'test_score': test_score,
			'coef': coef,
			'train_preds': train_preds,
			'test_preds': test_preds,
			'train_prob_preds':train_prob_preds,
			'test_prob_preds': test_prob_preds,
			'train_roc_auc': train_roc_auc,
			'test_roc_auc':test_roc_auc,
			'accuracy': accuracy
			}
			
		#saving output
		if out_save:
			if verbose==True: print('Saving...')
			filename = s_prefix + str(model).split('(')[0]
			file = open(filename + '.txt','w')
			
			non_df = dict((k, output[k]) for k in ['train_score','test_score','train_roc_auc','test_roc_auc', 'accuracy'])
			file.write(repr(non_df))
			file.close()
			
			df_like = ['X_train','X_test','y_train','y_test',\
			'train_CM','test_CM',\
			'train_fpr','train_tpr',\
			'train_preds','train_prob_preds',\
			'test_fpr','test_tpr',\
			'test_preds','test_prob_preds',\
			'coef']
			if coef == False: 
				df_like.remove('coef')

			dfs = dict((k, output[k]) for k in df_like)
			for p in dfs.keys():
				pd.DataFrame(dfs[p]).to_csv(filename + '_' + p + '.csv')

		#end and return
		if verbose==True: print('Finished!')
		return output
			
	#Get ROC Curve
	def get_roc(fitted_model, X,y, mono=False):	
		# Generate the prediction values for each of the test observations using predict_proba() function rather than just predict
		#prob_preds = getattr(fitted_model, 'predict_proba')(X)[:,1]
		if mono==False:
			prob_preds = fitted_model.predict_proba(X)[:,1]
		else:
			prob_preds = fitted_model.predict_proba(X)
		# Store the false positive rate(fpr), true positive rate (tpr) in vectors
		fpr, tpr, _ = metrics.roc_curve(y, prob_preds)
		# Store the Area Under the Curve (AUC)
		roc_auc = metrics.auc(fpr, tpr)
		
		output = {
			'fpr':fpr,
			'tpr':tpr,
			'prob_preds': prob_preds,
			'roc_auc':roc_auc,
			}
			#fpr, tpr, prob_preds, roc_auc
		return output

	def feature_selector_comp(model, X, y, verbose=True, super_verbose=False, rfe = True, test_size = None, random_state=None):
		'''performs a meta comparison between feature selectors and selects the best
		'''
		#Initial
		if verbose: print('Performing Initial Fit')
		X_train, X_test, y_train ,y_test = model_selection.train_test_split(X,y, test_size = test_size, random_state=random_state)
		X0_fit = model.fit(X_train,y_train)
		test_scores0 = model.score(X_test,y_test)
		if verbose: print('Score %s' %test_scores0)
		#FFS
		X1, X1_train, X1_test, y1_train, y1_test, scores1 = modelling.full_vs_ffs(model,X,y, verbose=verbose, super_verbose=super_verbose)
		#RFE
		if rfe:
			X2, X2_train, X2_test, y2_train, y2_test, scores2 = modelling.full_vs_rfe(model,X,y, verbose=verbose)
		else:
			scores2 = 0	
		#SFS- forward
		X3, X3_train, X3_test, y3_train, y3_test, scores3 = modelling.full_vs_sfs(model,X,y, verbose=verbose, direction='forward')
		#SFS- backward
		X4, X4_train, X4_test, y4_train, y4_test, scores4 = modelling.full_vs_sfs(model,X,y, verbose=verbose, direction='backward')

		scores_list = [scores1, scores2, scores3, scores4]
		winner = max(scores_list)

		if test_scores0 >= winner:
			name = 'Original Set'
			print('Winner:' ,name)
			return X, X_train, X_test, y_train ,y_test, test_scores0, name
		elif scores1 == winner:
			name = 'FFS'
			print('Winner:' ,name)
			return X1, X1_train, X1_test, y1_train, y1_test, scores1, name
		elif scores2 == winner:
			name = 'RFE'
			print('Winner:' ,name)
			return X2, X2_train, X2_test, y2_train, y2_test, scores2, name
		elif scores3 == winner:
			name = 'SFS-FWD'
			print('Winner:' ,name)
			return X3, X3_train, X3_test, y3_train, y3_test, scores3, name
		else:
			name = 'SFS-BACK'
			print('Winner:' ,name)
			return X4, X4_train, X4_test, y4_train, y4_test, scores4, name
					
	#RFE Feature Selection Comparator
	def full_vs_rfe(model, X, y, X0_score=None, test_size=None, random_state=None, verbose=False):
		
		#perform a X0 fit/eval
		X_train, X_test, y_train ,y_test = model_selection.train_test_split(X,y, test_size = test_size, random_state=random_state)
		
		if X0_score != None:
			if verbose: print('Scores found. Skipping Initial Fit')
			X0_score = X0_score
		else:	
			if verbose: print('Fitting X0...')
			X0_fit = model.fit(X_train,y_train)
			X0_score = model.score(X_test,y_test)

		#define estimator and selector
		if verbose == True: print('Performing RFE...\n')
		estimator = model.fit(X_train,y_train)
		selector = feature_selection.RFE(estimator, verbose=1)
		fit_sel = selector.fit(X_train,y_train)

		#transform X
		X1 = X.loc[:,fit_sel.support_]

		#perform a X1 fit/eval
		X1_train, X1_test, y1_train ,y1_test = model_selection.train_test_split(X1,y, test_size = test_size, random_state=random_state)
		if verbose == True: print('Fitting Alternate X1...\n')
		X1_fit = model.fit(X1_train,y1_train)
		X1_score = model.score(X1_test,y1_test)

		print('\n')
		#comparison and selection
		if X1_score > X0_score:
			print('RFE DOES improve. Retaining X1')
			X_train, X_test, y_train, y_test = model_selection.train_test_split(X1,y, test_size=test_size, random_state=random_state)
			print('Original Score: %s' %X0_score)
			print('Post RFE Score: %s' %X1_score)
			print('Gain/Loss %s' %(X1_score - X0_score))
			X = X1
			score = X1_score
			del X1
		else:
			print('RFE does NOT improve. Retaining all features')
			print('Original Score: %s' %X0_score)
			print('Post RFE Score: %s' %X1_score)
			print('Gain/Loss %s' %(X1_score - X0_score))
			score = X0_score
			X = X
			del X1


		return X, X_train, X_test, y_train ,y_test, score

	#SFS Feature Selection Comparator
	def full_vs_sfs(model, X, y, X0_score = None, test_size=0.2, random_state=20, 
	n_features=None, direction='forward', verbose=False):
		
		#perform a X0 fit/eval
		X_train, X_test, y_train ,y_test = model_selection.train_test_split(X,y, test_size = test_size, random_state=random_state)
		
		if X0_score != None:
			if verbose: print('Scores found. Skipping Initial Fit')
			X0_score = X0_score
		else:	
			if verbose: print('Fitting X0...')
			X0_fit = model.fit(X_train,y_train)
			X0_score = model.score(X_test,y_test)

		#define Selector
		if verbose == True: print('Performing SFS '+direction+'...\n')
		sfs = feature_selection.SequentialFeatureSelector(model, n_features_to_select=n_features, direction=direction)
		fit_sel = sfs.fit(X_train,y_train)

		#Transform X
		X1 = X.loc[:,fit_sel.support_]
		
		#eval with X1
		X1_train, X1_test, y1_train ,y1_test = model_selection.train_test_split(X1,y, test_size = test_size, random_state=random_state)
		if verbose == True: print('Fitting Alternate X1...\n')
		X1_fit = model.fit(X1_train,y1_train)
		X1_score = model.score(X1_test,y1_test)
			
		#comparison and selection
		if X1_score > X0_score:
			print('SFS DOES improve. Retaining X1')
			X_train, X_test, y_train, y_test = model_selection.train_test_split(X1,y, test_size=test_size, random_state=random_state)
			print('Original Score: %s' %X0_score)
			print('Post SFS Score: %s' %X1_score)
			print('Gain/Loss %s' %(X1_score - X0_score))
			X = X1
			score = X1_score
			del X1
		else:
			print('SFS does NOT improve. Retaining all features')
			print('Original Score: %s' %X0_score)
			print('Post SFS Score: %s' %X1_score)
			print('Gain/Loss %s' %(X1_score - X0_score))
			score = X0_score
			X = X
			del X1

		return X, X_train, X_test, y_train ,y_test, score

	#Full/FFS Testing
	def full_vs_ffs(model,X,y, X0_score = None, verbose=False, super_verbose = False, 
	test_size = 0.2, random_state = 20):
		
		#perform a X0 fit/eval
		if verbose: print('Splitting X0...')
		X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=test_size, random_state=random_state)
			
		if X0_score != None:
			if verbose: print('Scores found. Skipping Initial Fit')
			X0_score = X0_score
		else:	
			if verbose: print('Fitting X0...')
			X0_fit = model.fit(X_train,y_train)
			X0_score = model.score(X_test,y_test)

		#Perform FFS
		if verbose: print('Performing FFS...\n')
		fwd_feat = modelling.fwd_feat_sel(model,X,y, super_verbose)
		if verbose: print('\n')

		#Updating X
		X1 = X[fwd_feat].copy()

		#perform a X1 fit/eval
		if verbose: print('Splitting X1...')
		X1_train, X1_test, y1_train, y1_test = model_selection.train_test_split(X1,y, test_size=test_size, random_state=random_state)
		
		if verbose: print('Fitting X1...')
		X1_fit = model.fit(X1_train,y1_train)
		X1_score = model.score(X1_test,y1_test)
				
		#comparison and selection
		if X1_score > X0_score:
			print('\n Forward Feature Selection DOES improve; Retaining X1')
			X_train, X_test, y_train, y_test = model_selection.train_test_split(X1,y, test_size=test_size, random_state=random_state)
			print('Original Score: %s' %X0_score)
			print('Post FFS Score: %s' %X1_score)
			print('Gain/Loss %s' %(X1_score - X0_score))
			X = X1.copy()
			score = X1_score
			del X1
		else:
			print('\n Forward Feature Selection does NOT improve; Retaining all features')
			print('Original Score: %s' %X0_score)
			print('Post FFS Score: %s' %X1_score)
			print('Gain/Loss %s' %(X1_score - X0_score))
			score= X0_score
			del X1

		return X, X_train, X_test, y_train, y_test, score

	#Forward Feature Selection
	def fwd_feat_sel(model,X, y,verbose=False):
		### DEFINING FEATURES USING FWD FEATURE SELECTION
		## Use Forward Feature Selection to pick a good model
		# start with no predictors
		included = []
		
		r2_list = []
		a_r2_list = []
		# keep track of model and parameters
		best = {'feature': '', 'r2': 0, 'a_r2': 0}

		# get the number of cases in the training data
		n = X.shape[0]

		while True:
			changed = False
			
			if verbose: 
				print('') 
			# list the features to be evaluated
			excluded = list(set(X.columns) - set(included))
			if verbose: 
				print('(Step) Excluded = %s' % ', '.join(excluded))  

			# for each remaining feature to be evaluated
			for new_column in excluded:	
				if verbose:
					print('(Step) Trying %s...' % new_column)
					print('(Step) - Features = %s' % ', '.join(included + [new_column]))

				# fit the model with the Training data
				fit = model.fit(X[included + [new_column]] , y)
				# calculate the score
				r2 = model.score(X[included + [new_column]] , y)
				# number of predictors in this model
				k = len(included) + 1
				# calculate the adjusted R^2
				adjusted_r2 = 1 - (((1-r2)*(n-1))/(n-k-1)) # calculate the Adjusted R^2

				if verbose: 
					print('(Step) - Adjusted R^2: This = %.3f; Best = %.3f' % (adjusted_r2, best['a_r2']))

				# if model improves
				if adjusted_r2 > best['a_r2']:
					# record new parameters
					best = {'feature': new_column, 'r2': r2, 'a_r2': adjusted_r2}
					# flag that found a better model
					changed = True
					if verbose:
						print('(Step) - New Best! : Feature = %s; R^2 = %.3f; Adjusted R^2 = %.3f' % (best['feature'], best['r2'], best['a_r2']))
			# END for

			# if found a better model after testing all remaining features
			if changed:
				# update control details
				included.append(best['feature'])
				r2_list.append(r2)
				a_r2_list.append(adjusted_r2)
				excluded = list(set(excluded) - set(best['feature']))
				print('Added feature %-4s with R^2 = %.3f and adjusted R^2 = %.3f' %(best['feature'], best['r2'], best['a_r2']))
				
			else:
				# terminate if no better model
				break

		print('')
		print('Resulting features:')
		print(', '.join(included))

		output = {
			'included':included,
			'r2':r2,
			'r2_list':r2_list,
			'adjusted_r2': adjusted_r2,
			'a_r2_list':a_r2_list
		}	
		return output


	# ---- PLOTTING
class plotting:
	def __init__(self):
		pass

	def dec_reg(model, X, y, legend=2):
		plot_decision_regions(X = X.to_numpy(), y = y.to_numpy(), clf = model, legend = legend)	

	#plot ROC Curve
	def roc(model, fpr,tpr, roc_auc, figsize=[8,5], font_scale=1, save=False, ext='png',
	 s_prefix = '', ax=None, cmap=['green','navy'], context='notebook', lw=2):
		#capturing args
		c1 = cmap[0]
		c2 = cmap[1]

		#If not part of a Subplot (i.e Standalone)
		if ax == None:
			sns.set_context(context, font_scale = font_scale)
			plt.figure(figsize=figsize)
			plt.plot(fpr, tpr, color=c1, lw = lw, linestyle = '--', label = 'ROC curve (area = %0.2f)' % roc_auc)
			sns.lineplot(data=[0, 1], color =c2, lw = lw, linestyle = '-')
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('Receiver Operating Characteristic')
			plt.legend(loc = "lower right")
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.0])

			if save == True:
				utils.dir_check(s_prefix)
				name = s_prefix + str(model).split('(')[0] + '_roc_curve.' + ext
				plt.savefig(name)

		else: #part of a subplot
			sns.set_context('notebook', font_scale = font_scale)
			sns.set_theme(font_scale= font_scale)
			ax.set_title('ROC Curve', size = 21)
			df = pd.DataFrame(data={'False Positive Rate':fpr,'True Positive Rate':tpr})
			sns.lineplot(data=df, x='False Positive Rate', y='True Positive Rate', color=c1, lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc, ax=ax)
			sns.lineplot(data=[[0, 1]], color = c2, lw = lw, linestyle = '--', ax=ax)
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.0])
		
		return 
	
	# plot Confusion Matrix
	def confusion_matrix(model,confusion_matrix, figsize=[8,5], font_scale=1, save=False,
	 ext='png', s_prefix = '', ax=None, cmap=sns.diverging_palette(240,10)):

		#If not part of a subplot / standalone
		if ax==None:
			f, ax = plt.subplots(figsize=figsize)
			f.suptitle(str(model)+' Confusion Matrix')
			sns.set_theme(font_scale = font_scale)
			sns.set_context('notebook', font_scale = font_scale)
			ax = sns.heatmap(confusion_matrix, annot = True, fmt='d', cbar=False, cmap = cmap)

			#Save standalone png			
			if save == True:
				utils.dir_check(s_prefix)
				name = s_prefix + str(model).split('(')[0] + '_CM.' + ext
				plt.savefig(name)			
		else: #Part of a subplot
			sns.set_theme(font_scale = 1.5)
			sns.set_context('notebook', font_scale = font_scale)
			ax.set_title('Confusion Matrix', size=21)	
			sns.heatmap(confusion_matrix, annot = True,	fmt='d', cbar=False, cmap = cmap,ax=ax)
			
		#reset style
		sns.set_theme(font_scale = 1)	
		return

#plotting Multiple Models' ROC Curves on a single plot - DEPRECATED
		# def multi_roc(models, X,y):
		# 	colors = ['red','green','blue','yellow','black']
		# 	if len(models) < len(colors):
		# 		plt.figure(figsize=[18,10])
		# 		c = 0
		# 		for model in models:
		# 			print('plotting %s' %str(model))
		# 			globals()[f'predict_df_+{str(model)}'] = \
		# 				pd.DataFrame(model.predict_proba(X), columns=['class_0_pp','class_1_pp'])

		# 			globals()[f'fpr_+{str(model)}'], globals()[f'tpr_+{str(model)}'], _ = \
		# 				metrics.roc_curve(
		# 					y, 
		# 					globals()[f'predict_df_+{str(model)}']['class_1_pp']
		# 					)

		# 			globals()[f'roc_auc_+{str(model)}'] = \
		# 				metrics.auc(
		# 					globals()[f'fpr_+{str(model)}'], 
		# 					globals()[f'tpr_+{str(model)}']
		# 					)

		# 			plt.plot(
		# 				globals()[f'fpr_+{str(model)}'],
		# 				globals()[f'tpr_+{str(model)}'], 
		# 				color=colors[c], 
		# 				label=str(model).split('(')[0]+' ROC curve (area = %0.2f)' % globals()[f'roc_auc_+{str(model)}']
		# 				)
		# 			c +=1

		# 		plt.xlim([-0.05, 1.0])
		# 		plt.ylim([-0.05, 1.05])
		# 		plt.xlabel('False Positive Rate', fontsize=18)
		# 		plt.ylabel('True Positive Rate', fontsize=18)
		# 		plt.title('Receiver Operating Characteristic: M', fontsize=18)
		# 		plt.legend(loc="lower right")
		# 		plt.show()
		# 	else: 
		# 		print('ERROR: Too many models, max %s' %len(colors))

class utils:
	def __init__(self):
		pass

	def dir_check(dir):
		d = dir + '/'
		d = dir.split('/')
		levels=len(d)
		for level in range(levels):
			b = '/'.join(d[0:level+1])
			if level == levels:
				b += '/'
			if os.path.isdir(b)==False:
				os.mkdir(b)
				print('Directory created: %s' % b)	
			
			