from sklearn.linear_model import LogisticRegression
from sklearn.svc import SVC

def plot_weight(final_df,weights):
    #Visulaize the weights for different features
    error_y=dict(
        type='data',
        array=np.std(weights,axis=0),
        visible=True
    )

    graph1 = {'x': final_df.columns,
              'y': np.mean(weights,axis=0),
              'error_y':error_y,
              'type': 'bar'}

    fig = dict()
    fig['data'] = [graph1]
    fig['layout'] = {'title': 'Logistic Regression Weights, with error bars'}

    plotly.offline.iplot(fig)

def plot_metrics(metrics):
    metrics_df = pd.melt(metrics,id_vars=['Date','reg_param'],value_vars=['Accuracy','Train_Data_Size','Test_Data_Size',
                                                                          'Precision','Recall','F1_Score','Class_Ratio'],
                        var_name = 'Name',value_name='Metrics')
    g = sns.FacetGrid(metrics_df,col="reg_param",size = 4,aspect=0.5,hue = "Name")
    g.map(plt.plot,marker="o",x = "Date",y = "Metrics")
    g.fig.tight_layout(w_pad=1)

## Logistic Regression
reg_param = np.arange(0.0001,10,0.1)
metrics_df = pd.DataFrame()

for c in reg_param:
	clf_lr = LogisticRegression(penalty = 'l2',C = c,weight = 'balanced',warm_start = True,
	                            n_jobs = -1,solver = 'lbfgs')
	clf_perf = build_model(df,clf_lr)
	metrics = clf_perf[1]
	metrics = pd.DataFrame(metrics,columns = ['Date','Accuracy','Train_Data_Size','Test_Data_Size',
	                       'Precision','Recall','F1_Score','Class_Ratio'])
	metrics['reg_param'] = c
	weights = np.array(clf_perf[0].coef_[0])
	# weights = np.array(clf_perf[2])
	# plot_weight(weights)
	metrics_df = pd.concat((metrics_df,metrics))

## SVM
# Linear SVM

reg_param = np.arange(0.0001,10,0.1)
metrics_df = pd.DataFrame()

def rbfSVM(c):
	clf_rbfSVM = SVC(kernel = 'rbf',C=c,gamma = 'auto',weight = 'balanced',
	                    class_weight = 'balanced',random_state = 1)
	clf_perf = build_model(df,clf_rbfSVM)
	metrics = clf_perf[1]
	metrics = pd.DataFrame(metrics,columns = ['Date','Accuracy','Train_Data_Size','Test_Data_Size',
	                       'Precision','Recall','F1_Score','Class_Ratio'])
	metrics['reg_param'] = c
	return (metrics)

if __name__ == '__main__':
	num_cores = multiprocessing.cpu_count()
	metrics_df = Parallel(n_jobs = -1,backend = 'threading')(delayed(rbfSVM) for c in reg_param)

## MLP
from sklearn.neural_network import MLPClassifier

clf_mlp = MLPClassifier(hidden_layer_sizes=(50,),warm_start=True,activation = 'relu')
clf_perf = build_model(df,clf_mlp)
metrics = clf_perf[1]
metrics = pd.DataFrame(metrics,columns = ['Date','Accuracy','Train_Data_Size','Test_Data_Size',
	                       'Precision','Recall','F1_Score','Class_Ratio'])


## Random Forest

from sklearn.ensemble import RandomForestClassifier as rf

clf_rf = rf(n_estimators = 20,criterion='gini',max_features = 'auto',bootstrap = True,
            n_jobs = -1,warm_start = True,class_weight = True,random_state = 1)
clf_perf = build_model(df,clf_rf)
metrics = clf_perf[1]
metrics = pd.DataFrame(metrics,columns = ['Date','Accuracy','Train_Data_Size','Test_Data_Size',
	                       'Precision','Recall','F1_Score','Class_Ratio'])

















