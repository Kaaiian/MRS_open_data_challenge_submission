import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec


from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict, learning_curve, GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, classification_report, r2_score, mean_squared_error, auc, roc_curve, precision_recall_fscore_support

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def plot_1d_grid_search(grid, midpoint=0.7):

    mean_test_scores = grid.cv_results_['mean_test_score']
    parameter_name = list(grid.cv_results_['params'][0].keys())[0]
    parameters = grid.cv_results_['param_'+parameter_name]

    plt.figure(figsize=(6, 6))
    plt.plot(list(parameters), list(mean_test_scores), 'k--')
    plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
    plt.xlabel(parameter_name)
    plt.ylabel('R2 score')
    plt.title('grid search')


def plot_2d_grid_search(grid, midpoint=0.7, vmin=0, vmax=1):
    parameters = [x[6:] for x in list(grid.cv_results_.keys()) if 'param_' in x]

    param1 = list(set(grid.cv_results_['param_'+parameters[0]]))
    if parameters[1] == 'class_weight':
        param2 =list(set([d[1] for d in grid.cv_results_['param_'+parameters[1]]]))
    else:
        param2 =list(set(grid.cv_results_['param_'+parameters[1]]))
    scores = grid.cv_results_['mean_test_score'].reshape(len(param1),
                                                         len(param2))

    param1 = [round(param, 2) for param in param1]
    param2 = [round(param, 2) for param in param2]

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=midpoint))
    plt.xlabel(parameters[1])
    plt.ylabel(parameters[0])    
    plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
    plt.colorbar()
    plt.xticks(np.arange(len(param2)), sorted(param2), rotation=90)
    plt.yticks(np.arange(len(param1)), sorted(param1))

    plt.title('grid search')


def plot_prob(threshold, y_act, y_prob, threshold_x=300):

    color = ['#e41a1c','#377eb8','#4daf4a','#984ea3']

    y_act_labeled = [1 if x > threshold_x else 0 for x in y_act]
    y_pred_labeled = [1 if x >= threshold else 0 for x in y_prob]
    precision, recall, fscore, support = precision_recall_fscore_support(y_act_labeled, y_pred_labeled)
    print('precision: {:0.2f}\nrecall: {:0.2f}'.format(precision[1], recall[1]))

    tn, fp, fn, tp = confusion_matrix(y_act_labeled, y_pred_labeled).ravel()

    fig, ax = plt.subplots(1, figsize=(6, 6))
    rect1 = patches.Rectangle((threshold_x,-0.02), 600, threshold + 0.02, linewidth=1, edgecolor='k', facecolor=color[3], alpha=0.4, label='false negative ({:0.0f})'.format(fn))
    rect2 = patches.Rectangle((-50, threshold), threshold_x+50, 600, linewidth=1, edgecolor='k', facecolor=color[1], alpha=0.4, label='false postive ({:0.0f})'.format(fp))
    rect3 = patches.Rectangle((threshold_x, threshold), 600, 600, linewidth=1, edgecolor='k', facecolor=color[2], alpha=0.4, label='true positive ({:0.0f})'.format(tp))
    rect4 = patches.Rectangle((-50, -50), threshold_x+50, threshold+50, linewidth=1, edgecolor='k', facecolor='w', alpha=0.4, label='true negative ({:0.0f})'.format(tn))
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)

    plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
    plt.plot(y_act, y_prob, 'o', mfc='none', alpha=0.6, mec='#3F6077', mew=1.4)
    plt.plot([-10, 600], [threshold, threshold], 'k--', label='threshold', linewidth=3)

    plt.ylabel('Probability of being extraordinary')
    plt.xlabel('DFT computed bulk modulus')
    plt.xlim(-10, max(y_act)+50)
    plt.ylim(-.02, 1)

    plt.legend(loc=2, framealpha=0.4)
    return fig

def plot_regression(threshold, y_act, y_pred, threshold_x=300):

    color = ['#e41a1c','#377eb8','#4daf4a','#984ea3']

    y_act_labeled = [1 if x > threshold_x else 0 for x in y_act]
    y_pred_labeled = [1 if x >= threshold else 0 for x in y_pred]
    precision, recall, fscore, support = precision_recall_fscore_support(y_act_labeled, y_pred_labeled)
    print('precision: {:0.2f}\nrecall: {:0.2f}'.format(precision[1], recall[1]))

    tn, fp, fn, tp = confusion_matrix(y_act_labeled, y_pred_labeled).ravel()

    fig, ax = plt.subplots(1, figsize=(6, 6))
    rect1 = patches.Rectangle((threshold_x, -100), 600, threshold + 100, linewidth=1, edgecolor='k', facecolor=color[3], alpha=0.4, label='false negative ({:0.0f})'.format(fn))
    rect2 = patches.Rectangle((-50, threshold), threshold_x+50, 600, linewidth=1, edgecolor='k', facecolor=color[1], alpha=0.4, label='false postive ({:0.0f})'.format(fp))
    rect3 = patches.Rectangle((threshold_x, threshold), 600, 600, linewidth=1, edgecolor='k', facecolor=color[2], alpha=0.4, label='true positive ({:0.0f})'.format(tp))
    rect4 = patches.Rectangle((-50, -50), threshold_x+50, threshold+50, linewidth=1, edgecolor='k', facecolor='w', alpha=0.4, label='true negative ({:0.0f})'.format(tn))
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)
    
    plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
    plt.plot(y_act, y_pred, 'o', mfc='none', alpha=0.6, mec='#3F6077', mew=1.4)
    plt.plot([-10, 600], [threshold, threshold], 'k--', label='threshold', linewidth=3)

    plt.ylabel('Predicted bulk modulus')
    plt.xlabel('DFT computed bulk modulus')
    plt.xlim(-10, max(y_act)+50)
    plt.ylim(-10, max(y_act)+50)

#    plt.legend(loc=2)
    plt.legend(loc=2, framealpha=0.4)
    return fig
    

def plot_log_reg_grid_search(parameter_candidates, roc_means, fscore_means):
    parameters = list(parameter_candidates.values())
    param1, param2 = parameters[0], parameters[1]

    param1 = [float('{:.2f}'.format(x)) for x in param1]
    param2 = [float('{:.0f}'.format(x)) for x in param2]

    color_palette = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d']

    fig = plt.figure(1, figsize=(14, 8))

    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0.25, hspace=0.15)

    plot_roc = fig.add_subplot(gs[0:0, 0:1])
    plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
    data = np.array(roc_means).reshape(len(param1), len(param2))

    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(data, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=96, vmax=100, midpoint=98))
    plt.xlabel('class_weight')
    plt.ylabel('C')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(param2)), sorted(param2))
    plt.yticks(np.arange(len(param1)), sorted(param1))
    plt.title('AUC ROC')

    plot_fscore = fig.add_subplot(gs[0:0, 1:2])
    plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
    data = np.array(fscore_means).reshape(len(param1), len(param2))
    plt.subplots_adjust(left=0, right=0.99, bottom=0.15, top=0.95)
    plt.imshow(data, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=35, vmax=55, midpoint=45))
    plt.xlabel('class_weight')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(param2)), sorted(param2))
    plt.yticks(np.arange(len(param1)), sorted(param1))
    plt.title('F score')

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    inspired by:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve
                      .html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure(figsize=(6, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

def rf_feature_importance(rf, X_train, N='all', std_deviation=False):
    '''Get feature importances for trained random forest object
    
    Parameters
    ----------
    rf : sklearn RandomForest object
    	This needs to be a sklearn.ensemble.RandomForestRegressor of RandomForestClassifier object that has been fit to data
    N : integer, optional (default=10)
    	The N most important features are displayed with their relative importance scores
    std_deviation : Boolean, optional (default=False)
    	Whether or not error bars are plotted with the feature importance. (error can be very large if maximum_features!='all' while training random forest
    Output
    --------
    graphic :
    	return plot showing relative feature importance and confidence intervals
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> rf = RandomForestRegressor(max_depth=20, random_state=0)
    >>> rf.fit(X_train, y_train)
    >>> rf_feature_importance(rf, N=15)
    ''' 
    if N=='all':
    	N=X_train.shape[1]
    importance_dic = {}
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
    			 axis=0)
    indices = np.argsort(importances)[::-1]
    indices = indices[0:N]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(0, N):
    	importance_dic[X_train.columns.values[indices[f]]]=importances[indices[f]]
    	print(("%d. feature %d (%.3f)" % (f + 1, indices[f], importances[indices[f]])),':', X_train.columns.values[indices[f]])
    
    
    # Plot the feature importances of the forest
    plt.figure(figsize=(6,6))
    plt.title("Feature importances")
    if std_deviation == True:
    	plt.bar(range(0, N), importances[indices], color="r", yerr=std[indices], align="center")
    else:
    	plt.bar(range(0, N), importances[indices], color="r", align="center")
    plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
    plt.xticks(range(0, N), indices, rotation=90)
    plt.xlim([-1, N])
    return X_train.columns.values[indices]

def plot_act_vs_pred(y_actual, y_predicted):
    plt.figure(figsize=(6,6))
    plt.plot(y_actual, y_predicted, marker='o', mfc='none', color='#0077be', linestyle='none')
    plt.plot([min([min(y_actual), min(y_predicted)]), max([max(y_actual), max(y_predicted)])], [min([min(y_actual), min(y_predicted)]), max([max(y_actual), max(y_predicted)])], 'k--')
    plt.title("actual versus predicted values")
    plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
    limits = [min([min(y_actual), min(y_predicted)]), max([max(y_actual), max(y_predicted)])]
    plt.xlim(limits)
    plt.ylim(limits)
    plt.xlabel('actual')
    plt.ylabel('predicted')
    
def get_roc_auc(actual, probability, plot=False):
        fpr, tpr, tttt = roc_curve(actual, probability, pos_label=1)
        roc_auc = auc(fpr, tpr)
        if plot is True:
            plt.figure(2, figsize=(6, 6))
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([-0.02, 1.02])
            plt.ylim([-0.02, 1.02])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")

            plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
        return roc_auc
    
def get_performance_metrics(actual, predicted, probability, plot=False):

    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    roc_auc = get_roc_auc(actual, probability, plot=plot) * 100

    recall = tp / (fn+tp) * 100
    precision = tp / (tp+fp) * 100

    # print('precision: {:0.2f}, recall: {:0.2f}'.format(precision, recall))
    fscore = 2  * (recall * precision) / (recall + precision)
    # print('f-score: {:0.2f}, ROC_auc: {:0.2f}'.format(fscore, roc_auc))
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    return fscore, roc_auc

def log_reg_grid_search(X, y, parameter_candidates, n_cv=3):
    kf = KFold(n_splits=n_cv, shuffle=True, random_state=1)
    fscore_dict = {}
    roc_dict = {}
    fscore_means = {}
    roc_means = {}
    X_gs = X.reset_index(drop=True)
    y_gs = y.reset_index(drop=True)
    i = 0

    # print('--------initializing grid search --------')
    for val_0 in parameter_candidates['C']:
        for val_1 in parameter_candidates['class_weight']:
            model = LogisticRegression(C=val_0, class_weight={0:1, 1:val_1})
            fscore_list = []
            roc_area_list = []
            i += 1
            # print('-------grid_point: {:0.0f}/{:0.0f}-------'.format(i, len(parameter_candidates['C']) * len(parameter_candidates['class_weight'])))

            for train_index, test_index in kf.split(X):
                X_train, X_test = X_gs.iloc[train_index], X_gs.iloc[test_index]
                y_train, y_test = y_gs.iloc[train_index], y_gs.iloc[test_index]
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                y_test_prob_both = model.predict_proba(X_test)
                y_test_prob = [probability[1] for probability in y_test_prob_both]
                fscore, roc_area = get_performance_metrics(y_test, y_test_pred, y_test_prob)
                fscore_list.append(fscore)
                roc_area_list.append(roc_area)
            fscore_dict[(val_0, val_1)] = fscore_list
            roc_dict[(val_0, val_1)] = roc_area_list
            fscore_means[(val_0, val_1)] = np.array(fscore_list).mean()
            roc_means[(val_0, val_1)] = np.array(roc_area_list).mean()

    return fscore_dict, roc_dict

